# shellcheck shell=bash
# Shared functions of the publication gate. Source this file. Do not run it.
#
# Callers: pre-push, publish.sh, scan-dir.sh and test/run-tests.sh.
# The caller sets these variables before it calls the scan functions:
#   DMK_PUBLIC_CONFIG   path of gitleaks.toml (public rules)
#   DMK_ALLOWLIST       path of the file allow-list (optional)
# Everything else comes from the environment (see README.md).

DMK_EXIT_FOUND=42
DMK_DEFAULT_MAX_BYTES=5242880

# File types that must never reach a public repo. Matched on the lower-case base name.
DMK_BLOCKED_TYPES=(
  '*.gguf' '*.safetensors' '*.bin' '*.pt' '*.onnx' '*.nsys-rep' '*.ncu-rep'
  '*.sqlite' '*.sqlite3' '*.db' '*.pdf' '*.tar' '*.tar.*' '*.tgz' '*.zip' '*.7z'
)

# ---------------------------------------------------------------- output

# Print a line on stderr. Append it to the report file when one is set.
dmk_say() {
  printf '%s\n' "$*" >&2
  if [ -n "${DMK_REPORT_FILE:-}" ]; then
    printf '%s\n' "$*" >>"$DMK_REPORT_FILE" 2>/dev/null || true
  fi
}

# ---------------------------------------------------------------- temp files

dmk_init_tmp() {
  [ -n "${DMK_TMPDIR:-}" ] && return 0
  DMK_TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/dmk-gate.XXXXXX") || return 1
  chmod 700 "$DMK_TMPDIR"
}

dmk_cleanup() {
  if [ -n "${DMK_TMPDIR:-}" ] && [ -d "$DMK_TMPDIR" ]; then
    rm -rf -- "$DMK_TMPDIR"
  fi
}

# ---------------------------------------------------------------- rules

dmk_private_rules_path() {
  printf '%s\n' "${DMK_PRIVATE_RULES:-$HOME/.config/decision-model-kernels/gitleaks-private.toml}"
}

# Check the private rules file. The gate refuses to run without it.
dmk_require_private_rules() {
  local f mode
  f=$(dmk_private_rules_path)
  if [ ! -f "$f" ] || [ ! -r "$f" ]; then
    dmk_say "REFUSED: the private rules file is missing or not readable."
    dmk_say "  Expected a file at \$DMK_PRIVATE_RULES or at the default location in ~/.config/decision-model-kernels/."
    dmk_say "  The gate does not run without it. See tools/publish/README.md."
    return 1
  fi
  if ! grep -q '^\[\[rules\]\]' "$f"; then
    dmk_say "REFUSED: the private rules file holds no [[rules]] block."
    return 1
  fi
  if grep -q '^\[extend\]' "$f"; then
    dmk_say "REFUSED: the private rules file must not have an [extend] section. The gate adds it."
    return 1
  fi
  mode=$(stat -c '%a' "$f" 2>/dev/null || echo '?')
  if [ "$mode" != 600 ] && [ "$mode" != 400 ]; then
    dmk_say "WARNING: the private rules file has mode $mode. Set it to 0600."
  fi
  return 0
}

# Build the two gitleaks configurations in DMK_TMPDIR.
#   defaults.toml  default rules, public rules, private rules (default allowlists apply)
#   project.toml   public rules and private rules only (no default allowlists)
# The second set closes a gap: the default allowlists skip files such as SVG
# images and treat some path-like text as safe, which is wrong for privacy rules.
dmk_build_configs() {
  local pub=$DMK_PUBLIC_CONFIG priv
  dmk_init_tmp || return 1
  priv=$(dmk_private_rules_path)
  [ -f "$pub" ] || { dmk_say "REFUSED: public rules file not found: $pub"; return 1; }
  DMK_CFG_DEFAULTS=$DMK_TMPDIR/defaults.toml
  DMK_CFG_PROJECT=$DMK_TMPDIR/project.toml
  {
    printf "[extend]\npath = '%s'\n\n" "$pub"
    cat "$priv"
  } >"$DMK_CFG_DEFAULTS"
  {
    sed '/^\[extend\]$/,/^$/d' "$pub"
    printf '\n'
    cat "$priv"
  } >"$DMK_CFG_PROJECT"
  if grep -q '^\[extend\]' "$DMK_CFG_PROJECT"; then
    dmk_say "REFUSED: could not remove the [extend] block from the public rules."
    return 1
  fi
  if [ -n "${DMK_IGNORE_FILE:-}" ]; then
    [ -f "$DMK_IGNORE_FILE" ] || { dmk_say "REFUSED: DMK_IGNORE_FILE not found."; return 1; }
  else
    DMK_IGNORE_FILE=$DMK_TMPDIR/empty.ignore
    : >"$DMK_IGNORE_FILE"
  fi
  # The same template serves every scan: location and rule, never the text.
  printf '{{ range . }}{{ .File }}:{{ .StartLine }} {{ .RuleID }} {{ printf "%%.12s" .Commit }}\n{{ end }}' \
    >"$DMK_TMPDIR/report.tmpl"
  # Prove that gitleaks accepts both configurations before any real scan.
  local d=$DMK_TMPDIR/probe c rc
  mkdir -p "$d"
  for c in "$DMK_CFG_DEFAULTS" "$DMK_CFG_PROJECT"; do
    dmk_gl_run "$c" dir - "$d"
    rc=$?
    if [ "$rc" -ne 0 ]; then
      dmk_say "REFUSED: gitleaks cannot load the rules (check the private rules file syntax)."
      dmk_gl_tail
      return 1
    fi
  done
  return 0
}

# ---------------------------------------------------------------- gitleaks

dmk_gl_tail() {
  [ -f "$DMK_TMPDIR/gl.log" ] || return 0
  tail -n 5 "$DMK_TMPDIR/gl.log" | while IFS= read -r l; do dmk_say "    gitleaks: $l"; done
}

# Run gitleaks once. Return 0 (clean), 42 (findings) or 2 (error).
# Arguments: <config> <subcommand> <stdin file or -> [extra gitleaks arguments...]
# Findings go to $DMK_TMPDIR/gl.report as "file:line rule commit".
dmk_gl_run() {
  local config=$1 sub=$2 input=$3 rc
  shift 3
  : >"$DMK_TMPDIR/gl.report"
  local -a cmd=("${DMK_GITLEAKS:-gitleaks}" "$sub" --no-banner --no-color --redact --ignore-gitleaks-allow
    --exit-code "$DMK_EXIT_FOUND" --config "$config" --gitleaks-ignore-path "$DMK_IGNORE_FILE"
    --report-format template --report-template "$DMK_TMPDIR/report.tmpl"
    --report-path "$DMK_TMPDIR/gl.report" "$@")
  if [ "$input" = "-" ]; then
    "${cmd[@]}" </dev/null >"$DMK_TMPDIR/gl.log" 2>&1
  else
    "${cmd[@]}" <"$input" >"$DMK_TMPDIR/gl.log" 2>&1
  fi
  rc=$?
  case $rc in
    0) return 0 ;;
    "$DMK_EXIT_FOUND") return 42 ;;
    *) return 2 ;;
  esac
}

# Run both passes (default rules plus project rules, then project rules alone).
# The union of findings, without duplicates, lands in $DMK_TMPDIR/found.txt.
# Return 0 (clean), 1 (findings) or 2 (error).
dmk_gl_both() {
  local rc cfg any=0
  : >"$DMK_TMPDIR/found.raw"
  for cfg in "$DMK_CFG_DEFAULTS" "$DMK_CFG_PROJECT"; do
    dmk_gl_run "$cfg" "$@"
    rc=$?
    if [ "$rc" -eq 2 ]; then
      dmk_say "ERROR: gitleaks failed."
      dmk_gl_tail
      return 2
    fi
    if [ "$rc" -eq 42 ]; then any=1; fi
    cat "$DMK_TMPDIR/gl.report" >>"$DMK_TMPDIR/found.raw"
  done
  grep -v '^[[:space:]]*$' "$DMK_TMPDIR/found.raw" | sort -u >"$DMK_TMPDIR/found.txt" || true
  if [ -s "$DMK_TMPDIR/found.txt" ] || [ "$any" -eq 1 ]; then
    return 1
  fi
  return 0
}

# Print the findings of the last scan. Argument: a label.
# Optional second argument: a stream map file (see dmk_scan_meta).
dmk_print_found() {
  local label=$1 n shown=40
  n=$(grep -c . "$DMK_TMPDIR/found.txt" || true)
  DMK_FINDING_COUNT=$(( ${DMK_FINDING_COUNT:-0} + n ))
  dmk_say "  FINDINGS in $label: $n (text redacted)"
  head -n "$shown" "$DMK_TMPDIR/found.txt" | while IFS= read -r l; do dmk_say "    $l"; done
  if [ "$n" -gt "$shown" ]; then dmk_say "    ... and $((n - shown)) more"; fi
}

# Resolve the repository argument for gitleaks: the git directory, never the work tree.
# gitleaks reads a .gitleaksignore file from the source directory. A file in the
# work tree must not hide a finding, so the scan source is the git directory.
dmk_repo_path() {
  git -C "$1" rev-parse --absolute-git-dir
}

# Set DMK_BASE_ARGS to one ^<commit> argument for each revision in the upstream base.
# The base comes from $DMK_UPSTREAM_BASE, then from the git setting dmk.upstreamBase.
dmk_base_args() {
  local repo=$1 base w sha
  DMK_BASE_ARGS=()
  base=${DMK_UPSTREAM_BASE-}
  if [ -z "$base" ]; then base=$(git -C "$repo" config --get dmk.upstreamBase 2>/dev/null || true); fi
  for w in $base; do
    if ! sha=$(git -C "$repo" rev-parse --verify -q "$w^{commit}"); then
      dmk_say "REFUSED: the upstream base '$w' does not resolve to a commit in this repository."
      return 1
    fi
    DMK_BASE_ARGS+=("^$sha")
  done
  return 0
}

# Scan commit contents for the revisions in "$@" (a git rev-list argument list).
# Return 0 (clean), 1 (findings) or 2 (error).
dmk_scan_commits() {
  local repo=$1 label=$2 count rc
  shift 2
  count=$(git -C "$repo" rev-list --count "$@") || return 2
  if [ "$count" -eq 0 ]; then
    dmk_say "  $label: no new commits to scan"
    return 0
  fi
  dmk_gl_both git - "--log-opts=$*" "$(dmk_repo_path "$repo")"
  rc=$?
  if [ "$rc" -eq 1 ]; then
    dmk_print_found "$label: file contents of $count commit(s)"
  elif [ "$rc" -eq 0 ]; then
    dmk_say "  $label: $count commit(s), file contents clean"
  fi
  return "$rc"
}

# Write the text that becomes public with a push but is not scanned as a commit diff:
# commit messages, author and committer names and addresses, ref names, tag messages,
# and the lines that a merge commit adds in its conflict resolution.
# (gitleaks does not show the diff of a merge commit.)
# Arguments: <repo> <output file> <ref names file> <rev-list arguments...>
dmk_write_meta() {
  local repo=$1 out=$2 refs=$3 sha type name m
  shift 3
  {
    if [ -s "$refs" ]; then
      while IFS=' ' read -r sha name; do
        printf '@@ ref %s\n%s\n' "$name" "$name"
        type=$(git -C "$repo" cat-file -t "$sha" 2>/dev/null || true)
        if [ "$type" = tag ]; then
          printf '@@ tag message of %s\n' "$name"
          git -C "$repo" cat-file -p "$sha"
        fi
      done <"$refs"
    fi
    git -C "$repo" log --format='@@ commit %H%nauthor: %an <%ae>%ncommitter: %cn <%ce>%n%B' "$@"
    # A combined diff marks a line that no parent has with one plus sign for each parent.
    while IFS= read -r m; do
      printf '@@ merge resolution of %s\n' "$m"
      git -C "$repo" diff-tree --cc -p --no-commit-id "$m" | awk '
        /^diff / { inhunk = 0 }
        /^@@@+ / { inhunk = 1; n = length($1) - 1; next }
        inhunk { pre = substr($0, 1, n); gsub(/\+/, "", pre)
                 if (pre == "" && length($0) >= n) print substr($0, n + 1) }'
    done < <(git -C "$repo" rev-list --merges "$@")
  } >"$out"
}

# Scan the text from dmk_write_meta. Return 0 (clean), 1 (findings) or 2 (error).
dmk_scan_meta() {
  local meta=$1 label=$2 rc
  [ -s "$meta" ] || return 0
  dmk_gl_both stdin "$meta"
  rc=$?
  if [ "$rc" -eq 1 ]; then
    # Replace the line numbers by the owner of each line.
    awk 'FNR==NR {
           if ($0 ~ /^@@ /) { cur=$0; kind="" }
           else if (cur ~ /^@@ commit/) {
             if ($0 ~ /^author: /) kind="author"
             else if ($0 ~ /^committer: /) kind="committer"
             else kind="message"
           }
           own[FNR]=cur; typ[FNR]=kind; next }
         { split($1, a, ":"); n=a[2]; o=own[n]; if (o == "") o="stream"; k=typ[n]
           print "line " n " " $2 "  <- " o (k != "" ? " (" k ")" : "") }' \
      "$meta" "$DMK_TMPDIR/found.txt" >"$DMK_TMPDIR/found.mapped"
    sort -u "$DMK_TMPDIR/found.mapped" >"$DMK_TMPDIR/found.txt"
    dmk_print_found "$label: commit messages, identities, ref names and tag messages"
  elif [ "$rc" -eq 0 ]; then
    dmk_say "  $label: commit messages, identities and ref names clean"
  fi
  return "$rc"
}

# Scan a directory tree. Return 0 (clean), 1 (findings) or 2 (error).
dmk_scan_dir() {
  local dir=$1 rc
  dmk_gl_both dir - "$dir"
  rc=$?
  if [ "$rc" -eq 1 ]; then
    sed -i "s|^${dir%/}/||" "$DMK_TMPDIR/found.txt"
    dmk_print_found "$dir"
  fi
  return "$rc"
}

# ---------------------------------------------------------------- file checks

DMK_ALLOW=()

# Read the allow-list: one glob per line, "#" starts a comment.
dmk_load_allowlist() {
  local f=${1:-} line
  DMK_ALLOW=()
  [ -n "$f" ] && [ -f "$f" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%%#*}
    line=${line#"${line%%[![:space:]]*}"}
    line=${line%"${line##*[![:space:]]}"}
    if [ -n "$line" ]; then DMK_ALLOW+=("$line"); fi
  done <"$f"
}

dmk_path_allowed() {
  local p=$1 g
  for g in "${DMK_ALLOW[@]}"; do
    # shellcheck disable=SC2053
    if [[ $p == $g ]]; then return 0; fi
  done
  return 1
}

# Print the blocked-type pattern that matches a path. Return 1 when none matches.
dmk_blocked_type() {
  local b=${1##*/} g
  b=${b,,}
  for g in "${DMK_BLOCKED_TYPES[@]}"; do
    # shellcheck disable=SC2053
    if [[ $b == $g ]]; then printf '%s\n' "$g"; return 0; fi
  done
  return 1
}

# Check the files in the objects that the revisions in "$@" add.
# A file is blocked when it is larger than DMK_MAX_FILE_BYTES or has a blocked
# type, unless its path matches the allow-list. Return 0 (clean) or 1 (blocked).
# The check reads history, not only the current trees, because a file that was
# committed and later deleted is still public once the history is.
dmk_check_files() {
  local repo=$1 label=$2 max=${DMK_MAX_FILE_BYTES:-$DMK_DEFAULT_MAX_BYTES}
  shift 2
  case $max in ''|*[!0-9]*) dmk_say "REFUSED: DMK_MAX_FILE_BYTES is not a number."; return 2 ;; esac
  local -A size=() paths=()
  local type sha sz path tok meta o_mode n_mode o_sha n_sha status pat blocked=0 checked=0 p
  while IFS=' ' read -r type sha sz path; do
    [ "$type" = blob ] || continue
    size[$sha]=$sz
    paths[$sha]=$path
  done < <(git -C "$repo" rev-list --objects "$@" | git -C "$repo" cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)')
  # Add every other path that holds one of the new blobs.
  while IFS= read -r -d '' tok; do
    if [[ $tok == :* ]]; then
      meta=$tok
      IFS= read -r -d '' path || break
      read -r o_mode n_mode o_sha n_sha status <<<"$meta"
      if [ -n "${size[$n_sha]+x}" ] && [[ $'\n'${paths[$n_sha]}$'\n' != *$'\n'"$path"$'\n'* ]]; then
        paths[$n_sha]+=$'\n'"$path"
      fi
    fi
  done < <(git -C "$repo" log -m --no-renames --no-abbrev --diff-filter=AM --raw -z --format= "$@")
  local -a lines=()
  for sha in "${!size[@]}"; do
    checked=$((checked + 1))
    while IFS= read -r p; do
      [ -n "$p" ] || continue
      if dmk_path_allowed "$p"; then continue; fi
      if [ "${size[$sha]}" -gt "$max" ]; then
        lines+=("BLOCKED size: $p (${size[$sha]} bytes, limit $max)")
        blocked=1
      fi
      if pat=$(dmk_blocked_type "$p"); then
        lines+=("BLOCKED type: $p (matches $pat)")
        blocked=1
      fi
    done <<<"${paths[$sha]}"
  done
  if [ "$blocked" -eq 1 ]; then
    dmk_say "  FILE CHECK failed in $label:"
    printf '%s\n' "${lines[@]}" | sort -u | while IFS= read -r l; do dmk_say "    $l"; done
    DMK_FINDING_COUNT=$(( ${DMK_FINDING_COUNT:-0} + ${#lines[@]} ))
    return 1
  fi
  dmk_say "  $label: $checked new file version(s) checked, size and type clean"
  return 0
}

dmk_sha256() {
  sha256sum "$1" | cut -d' ' -f1
}

# Write a git credential helper to the path in $1. It answers a "get" request
# with the GitHub token from the environment variable GH_TOKEN at run time.
# The token is never written to the file and never appears in an argument.
dmk_write_gh_credential_helper() {
  printf '%s\n' '#!/bin/sh' '[ "$1" = get ] || exit 0' \
    'printf '"'"'username=x-access-token\npassword=%s\n'"'"' "$GH_TOKEN"' >"$1"
  chmod 700 "$1"
}
