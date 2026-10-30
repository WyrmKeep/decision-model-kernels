#!/usr/bin/env bash
# The publication gate. Run it on purpose, at the end of a phase or a pull request.
#
# Usage: publish.sh [--dry-run | --no-sync | --direct-push]
#
#   (no option)    scan, trigger the Gitea push-mirror sync, confirm GitHub equals Gitea
#   --dry-run      scan only. Nothing is pushed and no API call is made.
#   --no-sync      scan, then only compare the GitHub and Gitea heads and tags
#   --direct-push  scan, push to GitHub with git (fallback when Gitea cannot stop
#                  its periodic sync), then compare the heads and tags
#
# Steps
#   a  gitleaks over all refs (default rules, public rules, private rules)
#   b  size and type check over the history of all refs
#   c  sync trigger, or direct push
#   d  every GitHub head and tag must equal the Gitea one
# Steps c and d run only when steps a and b pass.
#
# The script is built to run under a credential helper that hides stdout and
# stderr. It writes a report without secrets to $DMK_REPORT and sets the exit code.
#
# Exit codes: 0 done, 1 scan blocked, 2 setup error, 3 sync failed,
#             4 GitHub differs from Gitea, 5 direct push failed.
#
# Environment (see README.md for the full list)
#   DMK_REPO              clone to publish (default: the clone that holds this script)
#   DMK_REPORT            report file (default: <git dir>/dmk-publish-report.txt)
#   DMK_PRIVATE_RULES     private rules file
#   DMK_UPSTREAM_BASE     upstream revisions to leave out of every scan
#   DMK_MAX_FILE_BYTES    file size limit (default 5242880)
#   GITEA_URL             Gitea base URL, for the API
#   GITEA_REPO            owner/name of the repository in Gitea
#   GITEA_ADMIN_USER      Gitea admin user name (basic authentication)
#   GITEA_ADMIN_PASSWORD  Gitea admin password
#   GITEA_GIT_URL         git URL of the Gitea repository (default: URL of remote origin)
#   GITHUB_URL            git URL of the GitHub repository (no credentials inside)
#   GH_TOKEN              GitHub token, for --direct-push
#   DMK_SYNC_TIMEOUT      seconds to wait for the mirror sync (default 180)
#   DMK_VERIFY_TIMEOUT    seconds to wait for equal heads (default 60)
set -u
export LC_ALL=C
export GIT_TERMINAL_PROMPT=0

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib.sh
. "$here/lib.sh"

dry=0 nosync=0 direct=0
for a in "$@"; do
  case $a in
    --dry-run) dry=1 ;;
    --no-sync) nosync=1 ;;
    --direct-push) direct=1 ;;
    -h|--help) sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "publish: unknown option $a" >&2; exit 2 ;;
  esac
done
if [ $((dry + nosync + direct)) -gt 1 ]; then
  echo "publish: use only one of --dry-run, --no-sync and --direct-push" >&2
  exit 2
fi

repo=${DMK_REPO:-}
if [ -z "$repo" ]; then repo=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || repo=; fi
if [ -z "$repo" ] || ! git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then
  echo "publish: cannot find the clone. Set DMK_REPO." >&2
  exit 2
fi
gitdir=$(git -C "$repo" rev-parse --absolute-git-dir)

DMK_REPORT=${DMK_REPORT:-$gitdir/dmk-publish-report.txt}
if ! ( umask 077; : >"$DMK_REPORT" ) 2>/dev/null; then
  echo "publish: cannot write the report file $DMK_REPORT" >&2
  exit 2
fi
DMK_REPORT_FILE=$DMK_REPORT

dmk_init_tmp || { echo "publish: cannot create a temporary directory" >&2; exit 2; }
trap dmk_cleanup EXIT

finish() {
  local code=$1 text=$2
  dmk_say "RESULT: $text (exit $code)"
  exit "$code"
}

mode=sync
[ "$dry" -eq 1 ] && mode=dry-run
[ "$nosync" -eq 1 ] && mode=no-sync
[ "$direct" -eq 1 ] && mode=direct-push
dmk_say "publication gate report"
dmk_say "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
dmk_say "mode: $mode"

# ---------------------------------------------------------------- setup checks

command -v "${DMK_GITLEAKS:-gitleaks}" >/dev/null 2>&1 || finish 2 "setup error: gitleaks is not installed or not in PATH (set DMK_GITLEAKS)"
if [ "$dry" -eq 0 ]; then
  command -v jq >/dev/null 2>&1 || [ "$mode" != sync ] || finish 2 "setup error: jq is not installed"
  if [ -z "${GITHUB_URL:-}" ]; then finish 2 "setup error: GITHUB_URL is not set"; fi
  if [[ ${GITHUB_URL} == *://*@* ]]; then
    finish 2 "setup error: GITHUB_URL holds credentials. Remove them. Use GH_TOKEN for --direct-push."
  fi
  if [ "$mode" = sync ]; then
    command -v curl >/dev/null 2>&1 || finish 2 "setup error: curl is not installed"
    for v in GITEA_URL GITEA_REPO GITEA_ADMIN_USER GITEA_ADMIN_PASSWORD; do
      [ -n "${!v:-}" ] || finish 2 "setup error: $v is not set"
    done
  fi
  if [ "$mode" = direct-push ] && [ -z "${GH_TOKEN:-}" ]; then
    finish 2 "setup error: GH_TOKEN is not set"
  fi
fi

dmk_require_private_rules || finish 2 "setup error: private rules file"
DMK_PUBLIC_CONFIG=$here/gitleaks.toml
DMK_ALLOWLIST=${DMK_ALLOWLIST:-$here/allow-files.txt}
if [ -z "${DMK_IGNORE_FILE:-}" ] && [ -f "$here/gitleaks.ignore" ]; then DMK_IGNORE_FILE=$here/gitleaks.ignore; fi
dmk_build_configs || finish 2 "setup error: rules"
dmk_load_allowlist "$DMK_ALLOWLIST"
dmk_base_args "$repo" || finish 2 "setup error: upstream base"

priv=$(dmk_private_rules_path)
dmk_say "public rules sha256: $(dmk_sha256 "$DMK_PUBLIC_CONFIG")"
dmk_say "private rules: sha256 $(dmk_sha256 "$priv"), $(grep -c '^\[\[rules\]\]' "$priv") rule blocks"
dmk_say "upstream base: ${DMK_UPSTREAM_BASE:-$(git -C "$repo" config --get dmk.upstreamBase 2>/dev/null || true)}"

# ---------------------------------------------------------------- Gitea refs

gitea_git=${GITEA_GIT_URL:-}
if [ -z "$gitea_git" ]; then gitea_git=$(git -C "$repo" config --get remote.origin.url 2>/dev/null || true); fi
if [ -n "$gitea_git" ]; then
  # The sync publishes what Gitea holds, so scan what Gitea holds, not only this clone.
  if ! git -C "$repo" fetch --quiet --prune --no-tags --no-recurse-submodules "$gitea_git" '+refs/*:refs/dmk-gitea/*' >"$DMK_TMPDIR/fetch.log" 2>&1; then
    dmk_say "ERROR: cannot fetch the refs from Gitea."
    tail -n 3 "$DMK_TMPDIR/fetch.log" | sed -E 's#(://)[^/@[:space:]]*@#\1***@#g' | while IFS= read -r l; do dmk_say "    git: $l"; done
    finish 2 "setup error: fetch from Gitea"
  fi
  dmk_say "fetched the Gitea refs into refs/dmk-gitea/ ($(git -C "$repo" for-each-ref refs/dmk-gitea | wc -l) refs)"
elif [ "$dry" -eq 1 ]; then
  dmk_say "NOTE: no Gitea URL. The dry run scans the local refs only."
else
  finish 2 "setup error: no Gitea git URL (set GITEA_GIT_URL or add remote origin)"
fi

# ---------------------------------------------------------------- steps a and b

DMK_FINDING_COUNT=0
blocked=0
dmk_say "step a: gitleaks over all refs"
dmk_scan_commits "$repo" "all refs" --all "${DMK_BASE_ARGS[@]}"
[ $? -eq 0 ] || blocked=1

git -C "$repo" for-each-ref --format='%(objectname) %(refname)' refs/heads refs/tags refs/dmk-gitea/heads refs/dmk-gitea/tags >"$DMK_TMPDIR/refs.txt"
dmk_write_meta "$repo" "$DMK_TMPDIR/meta.txt" "$DMK_TMPDIR/refs.txt" --all "${DMK_BASE_ARGS[@]}"
dmk_scan_meta "$DMK_TMPDIR/meta.txt" "all refs"
[ $? -eq 0 ] || blocked=1

dmk_say "step b: file size and type check over the history of all refs"
dmk_check_files "$repo" "all refs" --all "${DMK_BASE_ARGS[@]}"
[ $? -eq 0 ] || blocked=1

if [ "$blocked" -ne 0 ]; then
  dmk_say "No sync, no push and no API call was made."
  finish 1 "BLOCKED: $DMK_FINDING_COUNT item(s). Nothing was published"
fi
dmk_say "scans passed"
if [ "$dry" -eq 1 ]; then finish 0 "scans passed. Dry run, nothing published"; fi

# ---------------------------------------------------------------- step c

# Run git against GitHub with the token from the environment.
# The credential helper reads GH_TOKEN when git asks. The token is in no argument.
git_gh() {
  if [ -n "${GH_TOKEN:-}" ]; then
    git -c credential.helper= -c "credential.helper=!$DMK_TMPDIR/gh-credential" "$@"
  else
    git "$@"
  fi
}
if [ -n "${GH_TOKEN:-}" ]; then dmk_write_gh_credential_helper "$DMK_TMPDIR/gh-credential"; fi

# Call the Gitea API as admin. The login goes to curl on standard input, not in an argument.
# Prints the HTTP status. The body lands in $DMK_TMPDIR/api.body.
gitea_api() {
  local method=$1 url=$2 u p
  u=${GITEA_ADMIN_USER//\\/\\\\}; u=${u//\"/\\\"}
  p=${GITEA_ADMIN_PASSWORD//\\/\\\\}; p=${p//\"/\\\"}
  printf 'user = "%s:%s"\n' "$u" "$p" \
    | curl -sS -K - -X "$method" -H 'Accept: application/json' --max-time "${DMK_HTTP_TIMEOUT:-30}" \
        -o "$DMK_TMPDIR/api.body" -w '%{http_code}' "$url" 2>"$DMK_TMPDIR/api.err"
}

mirror_state() {
  local code
  code=$(gitea_api GET "${GITEA_URL%/}/api/v1/repos/${GITEA_REPO}/push_mirrors") || {
    dmk_say "  API call failed: $(sed -E 's#(://)[^/@[:space:]]*@#\1***@#g' "$DMK_TMPDIR/api.err" | tr '\n' ' ' | cut -c1-200)"
    return 1
  }
  if [ "$code" != 200 ]; then dmk_say "  GET push_mirrors answered HTTP $code"; return 1; fi
  jq -c 'map({n: (.remote_name // .remote_address // "?"), u: (.last_update // ""), e: (.last_error // "")}) | sort_by(.n)' \
    "$DMK_TMPDIR/api.body"
}

if [ "$mode" = sync ]; then
  dmk_say "step c: trigger the Gitea push-mirror sync"
  before=$(mirror_state) || finish 3 "sync failed: cannot read the push mirrors"
  if [ "$before" = '[]' ]; then finish 3 "sync failed: the repository has no push mirror"; fi
  code=$(gitea_api POST "${GITEA_URL%/}/api/v1/repos/${GITEA_REPO}/push_mirrors-sync") || code=000
  case $code in
    200|201|202|204) dmk_say "  sync requested (HTTP $code)" ;;
    *) finish 3 "sync failed: the sync request answered HTTP $code" ;;
  esac
  deadline=$((SECONDS + ${DMK_SYNC_TIMEOUT:-180}))
  verdict=wait
  while [ "$SECONDS" -lt "$deadline" ]; do
    sleep "${DMK_POLL_SECONDS:-3}"
    now=$(mirror_state) || continue
    verdict=$(jq -rn --argjson b "$before" --argjson n "$now" '
      ($b | map({(.n): .u}) | add) as $bu
      | [ $n[] | {n, moved: (.u != ($bu[.n] // "")), e} ] as $x
      | if ($x | any(.moved and (.e != ""))) then "error"
        elif ($x | all(.moved)) then "ok"
        else "wait" end')
    [ "$verdict" = wait ] || break
  done
  case $verdict in
    ok) dmk_say "  mirror sync finished without error" ;;
    error)
      dmk_say "  the mirror reports an error: $(jq -r '.[] | select(.e != "") | .e' <<<"$now" | sed -E 's#(://)[^/@[:space:]]*@#\1***@#g' | tr '\n' ' ' | cut -c1-300)"
      finish 3 "sync failed: the mirror reported an error"
      ;;
    *) finish 3 "sync failed: last_update did not move within ${DMK_SYNC_TIMEOUT:-180} s" ;;
  esac
elif [ "$mode" = direct-push ]; then
  dmk_say "step c: push to GitHub with git"
  # Push what Gitea holds, so GitHub equals Gitea. The plus signs allow a rewritten history.
  if git_gh -C "$repo" push --quiet --prune --no-verify "$GITHUB_URL" \
      '+refs/dmk-gitea/heads/*:refs/heads/*' '+refs/dmk-gitea/tags/*:refs/tags/*' >"$DMK_TMPDIR/push.log" 2>&1; then
    dmk_say "  push finished"
  else
    tail -n 4 "$DMK_TMPDIR/push.log" | sed -E 's#(://)[^/@[:space:]]*@#\1***@#g' | while IFS= read -r l; do dmk_say "    git: $l"; done
    finish 5 "direct push failed"
  fi
else
  dmk_say "step c: skipped (--no-sync)"
fi

# ---------------------------------------------------------------- step d

heads() { # <url> <use github credentials 0|1>
  if [ "$2" -eq 1 ]; then
    git_gh -C "$repo" ls-remote --heads --tags "$1"
  else
    git -C "$repo" ls-remote --heads --tags "$1"
  fi | sort
}

dmk_say "step d: GitHub heads and tags must equal the Gitea heads and tags"
deadline=$((SECONDS + ${DMK_VERIFY_TIMEOUT:-60}))
equal=0 gitea_ok=0 github_ok=0
while :; do
  gitea_ok=0 github_ok=0
  if gitea_set=$(heads "$gitea_git" 0 2>"$DMK_TMPDIR/ls1.err"); then gitea_ok=1; fi
  if github_set=$(heads "$GITHUB_URL" 1 2>"$DMK_TMPDIR/ls2.err"); then github_ok=1; fi
  if [ "$gitea_ok" -eq 1 ] && [ "$github_ok" -eq 1 ] && [ "$gitea_set" = "$github_set" ]; then equal=1; break; fi
  [ "$SECONDS" -lt "$deadline" ] || break
  sleep "${DMK_POLL_SECONDS:-3}"
done
if [ "$equal" -ne 1 ]; then
  if [ "$gitea_ok" -ne 1 ] || [ "$github_ok" -ne 1 ]; then
    dmk_say "  cannot read the remote refs: $(cat "$DMK_TMPDIR/ls1.err" "$DMK_TMPDIR/ls2.err" 2>/dev/null | sed -E 's#(://)[^/@[:space:]]*@#\1***@#g' | tr '\n' ' ' | cut -c1-200)"
  else
    dmk_say "  refs that differ (Gitea against GitHub):"
    diff <(printf '%s\n' "$gitea_set") <(printf '%s\n' "$github_set") | grep '^[<>]' | awk '{print "    " ($1=="<" ? "only or different in Gitea: " : "only or different in GitHub: ") $3}' | sort -u | head -n 40 | while IFS= read -r l; do dmk_say "$l"; done
  fi
  finish 4 "GitHub differs from Gitea"
fi
dmk_say "  $(printf '%s\n' "$gitea_set" | grep -vc '\^{}$') head and tag refs equal in both"
finish 0 "published. GitHub heads and tags equal Gitea"
