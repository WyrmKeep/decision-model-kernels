#!/usr/bin/env bash
# Tests for the publication gate.
#
# Usage: test/run-tests.sh [--keep]
#
# The tests need git, gitleaks, jq, curl and python3 (for the fake Gitea API).
# They build throwaway repositories under /tmp and use a test-only private
# rules file (private-rules.example.toml, with invented names).
# They never touch a network host. Local bare repositories stand in for Gitea
# and GitHub, and a fake HTTP server stands in for the Gitea API.
# Test data that looks private is built from parts at run time, so this file
# itself stays clean when the gate scans it.
# Exit code: 0 when every test passes.
set -u
export LC_ALL=C

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
P=$(cd "$here/.." && pwd -P)
keep=0
[ "${1:-}" = "--keep" ] && keep=1

for tool in git gitleaks jq curl python3 sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || { echo "run-tests: $tool is required" >&2; exit 2; }
done
real_git=$(command -v git)
real_curl=$(command -v curl)

T=$(mktemp -d /tmp/dmk-publish-test.XXXXXX)
fake_pids=()
cleanup() {
  local p
  for p in "${fake_pids[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
  if [ "$keep" -eq 1 ]; then echo "kept: $T"; else rm -rf "$T"; fi
}
trap cleanup EXIT

# An isolated environment: no user git settings, no host identity.
mkdir -p "$T/home" "$T/emptyhome" "$T/bin"
export HOME="$T/home"
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
export GIT_AUTHOR_NAME="Test Author" GIT_AUTHOR_EMAIL="author@example.invalid"
export GIT_COMMITTER_NAME="Test Author" GIT_COMMITTER_EMAIL="author@example.invalid"
export GIT_TERMINAL_PROMPT=0
unset DMK_UPSTREAM_BASE DMK_MAX_FILE_BYTES DMK_ALLOWLIST DMK_IGNORE_FILE DMK_REPORT DMK_REPO DMK_GITLEAKS
cp "$P/private-rules.example.toml" "$T/private.toml"
chmod 600 "$T/private.toml"
export DMK_PRIVATE_RULES="$T/private.toml"

pass=0 fail=0
fails=()
OUT=$T/out.txt
PENV=()
N=0

section() { printf '\n== %s ==\n' "$1"; }
ok() { printf 'PASS  %s\n' "$1"; pass=$((pass + 1)); }
bad() { printf 'FAIL  %s%s\n' "$1" "${2:+  ($2)}"; fail=$((fail + 1)); fails+=("$1"); }

# ---------------------------------------------------------------- data made from parts

ip4() { printf '%s.%s.%s.%s' "$1" "$2" "$3" "$4"; }
tsname() { printf 'node.tail0000.ts'; printf '.net'; }
uuid() { printf '%s-%s-%s-%s-%s' 11111111 2222 4333 8444 555555555555; }
kuma() { printf 'Kuma '; printf '#33'; }
mon() { printf 'monitor '; printf '#19'; }
srvpath() { printf '/%s/%s/%s' "$1" "$2" "$3"; }
secret_name() { printf 'example'; printf 'host'; }
owner_name() { printf 'wyrm'; printf 'keep'; }
fake_pat() { printf 'ghp_'; tr -dc 'A-Za-z0-9' </dev/urandom | head -c 36; }

# ---------------------------------------------------------------- repository helpers

# A bare remote and a clone with the hook installed. Sets R and W.
fresh() {
  N=$((N + 1))
  D=$T/case$N
  R=$D/remote.git
  W=$D/work
  mkdir -p "$D"
  git init -q --bare -b main "$R"
  git init -q -b main "$W"
  git -C "$W" remote add origin "$R"
  "$P/install-hooks.sh" "$W" >"$D/install.log" 2>&1
}

commit() { # <file> <content> [message]
  mkdir -p "$(dirname "$W/$1")"
  printf '%s\n' "$2" >"$W/$1"
  git -C "$W" add -A
  git -C "$W" commit -q -m "${3:-add $1}"
}

gpush() { ( cd "$W" && env "${PENV[@]}" git push "$@" ) >"$OUT" 2>&1; }

remote_state() { git -C "$R" for-each-ref --format='%(refname) %(objectname)'; }

expect_allowed() { # <name> [push arguments]
  local name=$1
  shift
  [ $# -gt 0 ] || set -- origin HEAD:refs/heads/main
  gpush "$@"
  local rc=$?
  if [ $rc -eq 0 ]; then ok "$name"; else bad "$name" "push refused, rc=$rc: $(tail -n 4 "$OUT" | tr '\n' ' ')"; fi
}

expect_refused() { # <name> <pattern in output> [push arguments]
  local name=$1 pat=$2 before after
  shift 2
  [ $# -gt 0 ] || set -- origin HEAD:refs/heads/main
  before=$(remote_state)
  gpush "$@"
  local rc=$?
  after=$(remote_state)
  if [ $rc -eq 0 ]; then bad "$name" "push was allowed"
  elif ! grep -q 'PUSH REFUSED' "$OUT"; then bad "$name" "refused without the gate message: $(tail -n 3 "$OUT" | tr '\n' ' ')"
  elif ! grep -Eq -- "$pat" "$OUT"; then bad "$name" "message lacks /$pat/: $(tail -n 5 "$OUT" | tr '\n' ' ')"
  elif [ "$before" != "$after" ]; then bad "$name" "the remote changed"
  else ok "$name"; fi
}

expect_absent() { # <name> <text that must not appear in the last output>
  if grep -qF -- "$2" "$OUT"; then bad "$1" "the output shows the text"; else ok "$1"; fi
}

# ================================================================ hook tests

section "hook: clean commits and allowed forms"
fresh
commit notes.md "Plain text about kernels, tokens and quantization."
expect_allowed "a clean commit passes"

fresh
commit docs.md "Documentation addresses are fine: $(ip4 192 0 2 10), $(ip4 198 51 100 7) and $(ip4 203 0 113 9)."
expect_allowed "documentation ranges pass (three RFC 5737 networks)"

fresh
commit edge.md "Near misses: $(ip4 100 63 0 1) $(ip4 100 128 0 1) $(ip4 172 15 0 1) $(ip4 172 32 0 1) $(ip4 11 0 0 1) version 1.10.0.1.2"
expect_allowed "addresses just outside the private ranges pass"

fresh
commit run.log "$(ip4 10 34 039 564) I perplexity: 606.47 seconds per pass"
expect_allowed "a log time stamp with a leading zero (minutes.seconds.milliseconds.microseconds) is not an address"

fresh
commit edge2.md "Numbers above 255 are no address: $(ip4 10 12 300 001) $(ip4 10 0 0 256) $(ip4 172 16 0 300)"
expect_allowed "numbers above 255 or with leading zeros are not an address"

fresh
commit README.md "Source: https://github.com/WyrmKeep/decision-model-kernels"
expect_allowed "the allowed GitHub owner URL passes"

fresh
commit LICENSE "MIT License

Copyright (c) 2026 WyrmKeep

Permission is hereby granted."
expect_allowed "the allowed license line passes"

section "hook: private addresses, names and paths block"
for spec in "10 0 0 1" "10 1 2 3" "10 255 255 255" "172 16 0 1" "172 16 0 9" "172 31 255 1" "192 168 1 5" "192 168 1 20" "100 64 0 1" "100 127 255 254"; do
  # shellcheck disable=SC2086
  set -- $spec
  fresh
  commit net.md "server at $(ip4 "$1" "$2" "$3" "$4") now"
  expect_refused "private address $(ip4 "$1" "$2" "$3" "$4") blocks" 'dmk-private-ipv4'
done

fresh
commit net.md "see $(tsname)"
expect_refused "a tailnet host name blocks" 'dmk-tailnet-hostname'

fresh
commit net.md "host $(secret_name) is down"
expect_refused "a private name in a file blocks" 'dmk-private-host'

fresh
commit item.md "item $(uuid)"
expect_refused "a UUID blocks" 'dmk-uuid'

for spec in "mnt user data" "mnt cache appdata" "boot config go" "var home someone" "home someone work" "root .ssh config"; do
  # shellcheck disable=SC2086
  set -- $spec
  fresh
  commit path.md "file $(srvpath "$1" "$2" "$3")/x"
  expect_refused "server path /$1/$2/ blocks" 'dmk-server-path'
done

fresh
commit mon.md "paused $(kuma) and $(mon) for the run"
expect_refused "an uptime monitor number blocks" 'dmk-monitor-number'

fresh
commit plot.svg "<svg><text>$(secret_name) run</text></svg>"
expect_refused "a private name inside an SVG file blocks (the default allowlists skip SVG)" 'dmk-private-host'

fresh
commit n.md "a bare owner name: $(owner_name)"
expect_refused "the owner name outside the allowed forms blocks" 'dmk-private-owner-name'

fresh
commit n.md "clone git@github.com:$(owner_name)/repo.git"
expect_refused "the owner name in an SSH-style URL blocks" 'dmk-private-owner-name'

fresh
commit n.md "see https://github.com/$(owner_name)/repo"
expect_refused "a lower-case owner URL blocks" 'dmk-private-owner-name'

fresh
commit LICENSE "Copyright (c) 2026 $(owner_name | sed 's/^./W/;s/keep$/Keep/') and contributors"
expect_refused "the license line with extra text blocks" 'dmk-private-owner-name'

section "hook: default gitleaks rules"
fresh
pat=$(fake_pat)
commit tok.env "TOKEN=$pat"
expect_refused "a GitHub-token-shaped secret blocks" 'github-pat'
expect_absent "the blocked output does not show the secret" "$pat"

section "hook: output stays redacted"
fresh
commit net.md "server at $(ip4 192 168 7 7) and host $(secret_name)"
expect_refused "findings print rule and location" 'net.md:1 dmk-'
expect_absent "the output does not show the address" "$(ip4 192 168 7 7)"
expect_absent "the output does not show the private name" "$(secret_name)"

section "hook: commit metadata, ref names and tag messages"
fresh
commit a.md "clean" "fix the mount on $(secret_name)"
expect_refused "a private name in a commit message blocks" '\(message\)'

fresh
( export GIT_AUTHOR_NAME="Dr $(secret_name)"; commit a.md "clean" )
expect_refused "a private name in an author name blocks" '\(author\)'

fresh
( export GIT_COMMITTER_EMAIL="me@$(secret_name).example.invalid"; commit a.md "clean" )
expect_refused "a private name in a committer address blocks" '\(committer\)'

fresh
commit a.md "clean"
expect_refused "a private name in a branch name blocks" '@@ ref' origin "HEAD:refs/heads/$(secret_name)-fix"

fresh
commit a.md "clean"
expect_allowed "a clean branch is pushed first"
git -C "$W" tag -a v1 -m "release for $(secret_name)"
expect_refused "a private name in an annotated tag message blocks" '@@ tag' origin v1

fresh
commit a.md "clean"
expect_allowed "a clean branch before the tag test"
git -C "$W" tag -a v2 -m "release 2"
expect_allowed "a clean annotated tag passes" origin v2

section "hook: only the new commits are scanned"
fresh
commit old.md "old address $(ip4 192 168 1 1)"
gpush --no-verify origin HEAD:refs/heads/main
commit new.md "clean change"
expect_allowed "an old finding already on the remote is not scanned again"
commit newer.md "new address $(ip4 10 9 9 9)"
expect_refused "a new finding on an existing branch blocks"  'dmk-private-ipv4'

fresh
commit one.md "one"
commit two.md "two $(ip4 10 0 0 7)"
gpush --no-verify origin HEAD:refs/heads/main
git -C "$W" checkout -q -b feature
commit three.md "three, clean"
expect_allowed "a new branch skips commits that a remote-tracking ref covers" origin HEAD:refs/heads/feature
grep -q '1 commit(s)' "$OUT" && ok "the new branch scan covers one commit" || bad "the new branch scan covers one commit" "$(grep -E 'commit\(s\)' "$OUT" | tr '\n' ' ')"

fresh
commit a.md "x"
commit b.md "y $(ip4 192 168 3 3)"
git -C "$W" tag light
expect_refused "a lightweight tag on a bad commit blocks" 'dmk-private-ipv4' origin light

section "hook: deletions pass"
fresh
commit a.md "clean"
expect_allowed "a clean branch before the delete test" origin HEAD:refs/heads/side
expect_allowed "a branch deletion passes" origin :refs/heads/side
PENV=(DMK_PRIVATE_RULES="$T/does-not-exist.toml")
fresh
commit a.md "clean"
gpush --no-verify origin HEAD:refs/heads/side
expect_allowed "a deletion passes even without the private rules file" origin :refs/heads/side
PENV=()

section "hook: the private rules file is required"
PENV=(DMK_PRIVATE_RULES="$T/does-not-exist.toml")
fresh
commit a.md "clean"
expect_refused "a missing private rules file refuses the push (DMK_PRIVATE_RULES)" 'private rules file is missing'
PENV=(DMK_PRIVATE_RULES= HOME="$T/emptyhome")
fresh
commit a.md "clean"
expect_refused "a missing private rules file refuses the push (default location)" 'private rules file is missing'
PENV=()
printf '[extend]\nuseDefault = true\n[[rules]]\nid = "x"\nregex = "y"\n' >"$T/bad-extend.toml"
chmod 600 "$T/bad-extend.toml"
PENV=(DMK_PRIVATE_RULES="$T/bad-extend.toml")
fresh
commit a.md "clean"
expect_refused "a private file with an [extend] section is refused" 'must not have an \[extend\]'
printf '[[rules]]\nid = "x"\nregex = "(unclosed"\n' >"$T/bad-regex.toml"
chmod 600 "$T/bad-regex.toml"
PENV=(DMK_PRIVATE_RULES="$T/bad-regex.toml")
fresh
commit a.md "clean"
expect_refused "a private file that gitleaks cannot load is refused" 'cannot load the rules'
PENV=()
printf '# no rules here\n' >"$T/empty.toml"
chmod 600 "$T/empty.toml"
PENV=(DMK_PRIVATE_RULES="$T/empty.toml")
fresh
commit a.md "clean"
expect_refused "a private file without rules is refused" 'no \[\[rules\]\]'
PENV=()
PENV=(DMK_GITLEAKS="$T/no-such-gitleaks")
fresh
commit a.md "clean"
expect_refused "a missing gitleaks program refuses the push" 'gitleaks is not installed'
PENV=()

section "hook: upstream history is not scanned with project rules"
mkfork() { # builds a fork-like history in W: upstream commits, a tag, one project commit
  fresh
  commit docs/net.md "upstream docs mention $(ip4 192 168 0 10)" "upstream: network docs"
  commit src/main.c "int main(void) { return 0; }" "upstream: code"
  git -C "$W" tag upstream-base
  commit project.md "project note, clean" "project: add note"
}
mkfork
PENV=(DMK_UPSTREAM_BASE=upstream-base)
expect_allowed "an address in upstream history is skipped when DMK_UPSTREAM_BASE is set"
PENV=()
mkfork
expect_refused "the same history blocks when DMK_UPSTREAM_BASE is not set" 'dmk-private-ipv4'
mkfork
"$P/install-hooks.sh" --upstream-base upstream-base "$W" >/dev/null 2>&1
expect_allowed "the git setting dmk.upstreamBase works like the variable"
mkfork
PENV=(DMK_UPSTREAM_BASE=upstream-base)
commit docs/net.md "upstream docs mention $(ip4 192 168 0 10) and now $(ip4 10 20 30 40)" "project: edit upstream file"
expect_refused "a project commit that adds an address to an upstream file still blocks" 'dmk-private-ipv4'
PENV=(DMK_UPSTREAM_BASE=no-such-revision)
mkfork
expect_refused "an upstream base that does not resolve refuses the push" 'does not resolve'
PENV=()

section "hook: merge commits"
merge_conflict() { # builds a merge whose conflict resolution is the text in $1
  fresh
  commit f.txt $'one\ntwo\nthree' "base"
  gpush --no-verify origin HEAD:refs/heads/main
  git -C "$W" checkout -q -b side
  commit f.txt $'one\nSIDE\nthree' "side change"
  git -C "$W" checkout -q main
  commit f.txt $'one\nMAIN\nthree' "main change"
  git -C "$W" merge -q side >/dev/null 2>&1
  printf 'one\n%s\nthree\n' "$1" >"$W/f.txt"
  git -C "$W" add -A
  git -C "$W" commit -q -m "merge side"
}
merge_conflict "resolved with $(ip4 192 168 4 4)"
expect_refused "a conflict resolution that adds an address blocks" 'merge resolution'
merge_conflict "resolved with plain words"
expect_allowed "a clean conflict resolution passes"

mkfork
git -C "$W" checkout -q -b newer upstream-base
commit docs/new.md "newer upstream docs mention $(ip4 10 50 50 50)" "upstream: newer docs"
git -C "$W" tag upstream-newer
git -C "$W" checkout -q main
git -C "$W" merge -q --no-edit upstream-newer >/dev/null 2>&1
PENV=(DMK_UPSTREAM_BASE="upstream-base")
expect_refused "a merge of newer upstream history blocks when only the old base is set" 'dmk-private-ipv4'
PENV=(DMK_UPSTREAM_BASE="upstream-base upstream-newer")
expect_allowed "the same merge passes when the newer upstream revision is in the base"
PENV=()

section "hook: size and type"
mkbig() { head -c "$2" /dev/zero >"$W/$1"; git -C "$W" add -A; git -C "$W" commit -q -m "add $1"; }
fresh
mkbig big.dat 6291456
expect_refused "a 6 MiB file blocks" 'BLOCKED size: big.dat'
fresh
mkbig four.dat 4194304
expect_allowed "a 4 MiB file passes"
fresh
mkbig exact.dat 5242880
expect_allowed "a file of exactly 5 MiB passes"
fresh
mkbig over.dat 5242881
expect_refused "a file one byte over 5 MiB blocks" 'BLOCKED size: over.dat'
fresh
mkbig mid.dat 2097152
PENV=(DMK_MAX_FILE_BYTES=1048576)
expect_refused "DMK_MAX_FILE_BYTES lowers the limit" 'BLOCKED size: mid.dat'
PENV=()
for ext in gguf safetensors bin pt onnx nsys-rep ncu-rep sqlite db pdf tar tar.gz zip 7z GGUF; do
  fresh
  printf 'tiny\n' >"$W/model.$ext"
  git -C "$W" add -A
  git -C "$W" commit -q -m "add model.$ext"
  expect_refused "a small .$ext file blocks by type" "BLOCKED type: model.$ext"
done
fresh
mkdir -p "$W/docs/figures"
head -c 6291456 /dev/zero >"$W/docs/figures/data.dat"
git -C "$W" add -A
git -C "$W" commit -q -m "add figure data"
expect_refused "a large file blocks when no allow-list entry covers it" 'BLOCKED size: docs/figures/data.dat'
printf 'docs/figures/*.dat   # test exception\n' >"$T/allow.txt"
PENV=(DMK_ALLOWLIST="$T/allow.txt")
expect_allowed "an allow-list entry lets a listed large file pass"
PENV=()
fresh
mkdir -p "$W/docs/figures"
head -c 6291456 /dev/zero >"$W/docs/figures/data.dat"
printf 'tiny\n' >"$W/docs/figures/summary.pdf"
git -C "$W" add -A
git -C "$W" commit -q -m "add figures"
PENV=(DMK_ALLOWLIST="$T/allow.txt")
expect_refused "an allow-list entry does not cover other files" 'BLOCKED type: docs/figures/summary.pdf'
PENV=()
fresh
mkbig gone.dat 6291456
git -C "$W" rm -q gone.dat
git -C "$W" commit -q -m "remove it again"
expect_refused "a large file that a later commit deletes still blocks (it stays in history)" 'BLOCKED size: gone.dat'

section "hook: gitleaks bypasses are closed"
fresh
commit a.md "server $(ip4 10 3 3 3) # gitleaks:allow"
expect_refused "an inline gitleaks:allow comment does not hide a finding" 'dmk-private-ipv4'
fresh
commit a.md "server $(ip4 10 4 4 4)"
fp="$(git -C "$W" rev-parse HEAD):a.md:dmk-private-ipv4:1"
printf '%s\n' "$fp" >"$W/.gitleaksignore"
expect_refused "a .gitleaksignore file in the repository is ignored" 'dmk-private-ipv4'
printf '%s\n' "$fp" >"$T/ignore.txt"
PENV=(DMK_IGNORE_FILE="$T/ignore.txt")
expect_allowed "DMK_IGNORE_FILE with the fingerprint lets that finding pass"
PENV=()

section "hook: install, checksums and integrity"
fresh
commit a.md "clean"
hook_sum=$(sha256sum "$W/.git/hooks/pre-push" | cut -d' ' -f1)
if grep -q "^$hook_sum  pre-push\$" "$W/.git/hooks/dmk-publish/SHA256SUMS"; then ok "install records the sha256 of the hook"; else bad "install records the sha256 of the hook"; fi
"$P/install-hooks.sh" --verify "$W" >/dev/null 2>&1 && ok "install-hooks --verify accepts the installed files" || bad "install-hooks --verify accepts the installed files"
printf '# edited\n' >>"$W/.git/hooks/dmk-publish/lib.sh"
"$P/install-hooks.sh" --verify "$W" >/dev/null 2>&1 && bad "install-hooks --verify rejects an edited file" || ok "install-hooks --verify rejects an edited file"
expect_refused "the hook refuses to run when a support file was edited" 'differ from their recorded checksums'
"$P/install-hooks.sh" "$W" >/dev/null 2>&1
expect_allowed "a reinstall repairs the hook"
fresh
rm -f "$W/.git/hooks/pre-push"
printf '#!/bin/sh\nexit 0\n' >"$W/.git/hooks/pre-push"
chmod +x "$W/.git/hooks/pre-push"
"$P/install-hooks.sh" "$W" >/dev/null 2>&1 && bad "install refuses to replace a foreign hook" || ok "install refuses to replace a foreign hook"
"$P/install-hooks.sh" --force "$W" >/dev/null 2>&1 && [ -f "$W/.git/hooks/pre-push.before-dmk-gate" ] && ok "install --force moves a foreign hook aside" || bad "install --force moves a foreign hook aside"
fresh
git -C "$W" config core.hooksPath /nonexistent
"$P/install-hooks.sh" "$W" >/dev/null 2>&1 && bad "install refuses when core.hooksPath is set" || ok "install refuses when core.hooksPath is set"

section "hook: the installed ignore file"
# A copy of the gate folder with its own ignore file, so these tests do not depend on the real one.
G=$T/gate-copy
mkdir -p "$G"
cp "$P/pre-push" "$P/install-hooks.sh" "$P/lib.sh" "$P/gitleaks.toml" "$P/allow-files.txt" "$G/"
fresh_gate() { # a fresh clone with the hook installed from the copy $G
  fresh
  "$G/install-hooks.sh" "$W" >"$D/install-gate.log" 2>&1
  ignore_installed=$W/.git/hooks/dmk-publish/gitleaks.ignore
}
ignored_line() { printf 'a.md:dmk-private-ipv4:1\n'; }

{ printf '# test ignore file\n'; ignored_line; } >"$G/gitleaks.ignore"
fresh_gate
[ -f "$ignore_installed" ] && cmp -s "$G/gitleaks.ignore" "$ignore_installed" && ok "install copies gitleaks.ignore to the hook folder" || bad "install copies gitleaks.ignore to the hook folder"
ignore_sum=$(sha256sum "$G/gitleaks.ignore" | cut -d' ' -f1)
if grep -q "^$ignore_sum  dmk-publish/gitleaks.ignore\$" "$W/.git/hooks/dmk-publish/SHA256SUMS"; then ok "install records the sha256 of the ignore file"; else bad "install records the sha256 of the ignore file"; fi
commit a.md "server $(ip4 10 4 4 4)"
expect_allowed "the hook uses the installed ignore file by default"

commit b.md "server $(ip4 10 5 5 5)"
expect_refused "the installed ignore file hides only the listed finding" 'b.md:1 dmk-private-ipv4'

# An explicit DMK_IGNORE_FILE wins over the installed copy.
fresh_gate
commit a.md "server $(ip4 10 4 4 4)"
printf 'b.md:dmk-private-ipv4:1\n' >"$T/other.ignore"
PENV=(DMK_IGNORE_FILE="$T/other.ignore")
expect_refused "an explicit DMK_IGNORE_FILE replaces the installed ignore file" 'a.md:1 dmk-private-ipv4'
PENV=()
expect_allowed "the same push passes with the installed ignore file"

# A .gitleaksignore file in the work tree stays without effect, with an installed ignore file in place.
fresh_gate
commit c.md "server $(ip4 10 6 6 6)"
printf '%s\n' "$(git -C "$W" rev-parse HEAD):c.md:dmk-private-ipv4:1" "c.md:dmk-private-ipv4:1" >"$W/.gitleaksignore"
expect_refused "a .gitleaksignore file in the repository stays ignored when an ignore file is installed" 'c.md:1 dmk-private-ipv4'

# Trust model: a change takes effect only after install-hooks.sh runs again.
fresh_gate
commit a.md "server $(ip4 10 4 4 4)"
printf 'a.md:dmk-private-ipv4:1\n' >>"$ignore_installed"
expect_refused "the hook refuses to run when the installed ignore file was edited" 'differ from their recorded checksums'
"$G/install-hooks.sh" --verify "$W" >/dev/null 2>&1 && bad "install-hooks --verify rejects an edited ignore file" || ok "install-hooks --verify rejects an edited ignore file"
"$G/install-hooks.sh" "$W" >/dev/null 2>&1
expect_allowed "a reinstall restores the installed ignore file"

fresh_gate
commit d.md "server $(ip4 10 7 7 7)"
printf 'd.md:dmk-private-ipv4:1\n' >>"$G/gitleaks.ignore"
expect_refused "a change in the repository ignore file has no effect before a reinstall" 'd.md:1 dmk-private-ipv4'
"$G/install-hooks.sh" "$W" >/dev/null 2>&1
expect_allowed "the change takes effect after install-hooks.sh runs again"

fresh_gate
commit a.md "server $(ip4 10 4 4 4)"
rm -f "$W/.git/hooks/dmk-publish/gitleaks.ignore"
expect_refused "the hook refuses to run when the installed ignore file is missing" 'differ from their recorded checksums'

# Without an ignore file in the gate folder, nothing is installed and a stray copy is not trusted.
rm -f "$G/gitleaks.ignore"
fresh_gate
[ ! -e "$W/.git/hooks/dmk-publish/gitleaks.ignore" ] && ok "install copies no ignore file when the folder has none" || bad "install copies no ignore file when the folder has none"
if grep -q 'gitleaks.ignore' "$W/.git/hooks/dmk-publish/SHA256SUMS"; then bad "install records no checksum for a missing ignore file"; else ok "install records no checksum for a missing ignore file"; fi
commit a.md "server $(ip4 10 4 4 4)"
ignored_line >"$W/.git/hooks/dmk-publish/gitleaks.ignore"
expect_refused "an ignore file without a recorded checksum is refused" 'has no recorded checksum'

# A reinstall from a folder without an ignore file removes the old copy.
{ printf '# test ignore file\n'; ignored_line; } >"$G/gitleaks.ignore"
fresh_gate
commit a.md "server $(ip4 10 4 4 4)"
rm -f "$G/gitleaks.ignore"
"$G/install-hooks.sh" "$W" >/dev/null 2>&1
[ ! -e "$ignore_installed" ] && ok "a reinstall removes an ignore file that the folder no longer has" || bad "a reinstall removes an ignore file that the folder no longer has"
expect_refused "the finding blocks again after the ignore file is gone" 'a.md:1 dmk-private-ipv4'

section "hook: other ways to run git"
fresh
commit a.md "x $(ip4 10 1 1 1)"
mkdir -p "$W/sub/dir"
( cd "$W/sub/dir" && git push origin HEAD:refs/heads/main ) >"$OUT" 2>&1 && bad "a push from a subdirectory runs the hook" "push was allowed" || { grep -q 'dmk-private-ipv4' "$OUT" && ok "a push from a subdirectory runs the hook" || bad "a push from a subdirectory runs the hook" "$(tail -n 3 "$OUT" | tr '\n' ' ')"; }
fresh
commit a.md "clean"
git -C "$W" worktree add -q "$D/wt" -b wtbranch
main_w=$W
W=$D/wt
commit w.md "x $(ip4 10 1 1 1)"
expect_refused "a push from a linked worktree runs the hook" 'dmk-private-ipv4' origin HEAD:refs/heads/wtbranch
W=$main_w
fresh
commit a.md "clean"
git -C "$W" push --dry-run origin HEAD:refs/heads/main >"$OUT" 2>&1 && ok "a clean dry-run push passes through the hook" || bad "a clean dry-run push passes through the hook"

# ================================================================ scan-dir tests

section "scan-dir"
mkdir -p "$T/dir-clean" "$T/dir-dirty"
printf 'Nothing private here.\n' >"$T/dir-clean/a.md"
printf '<svg><text>%s</text></svg>\n' "$(secret_name)" >"$T/dir-dirty/plot.svg"
"$P/scan-dir.sh" "$T/dir-clean" >/dev/null 2>&1 && ok "scan-dir passes a clean directory" || bad "scan-dir passes a clean directory"
"$P/scan-dir.sh" "$T/dir-dirty" >"$OUT" 2>&1 && bad "scan-dir blocks a private name in an SVG file" || { grep -q 'plot.svg:1 dmk-private-host' "$OUT" && ok "scan-dir blocks a private name in an SVG file" || bad "scan-dir blocks a private name in an SVG file" "$(cat "$OUT")"; }
DMK_PRIVATE_RULES="$T/does-not-exist.toml" "$P/scan-dir.sh" "$T/dir-clean" >/dev/null 2>&1 && bad "scan-dir refuses without the private rules file" || ok "scan-dir refuses without the private rules file"

# ================================================================ publish.sh tests

section "publish.sh scans (local bare repositories stand in for Gitea and GitHub)"

# Sets G (Gitea bare), H (GitHub bare) and W (clone with origin = G, one clean commit pushed).
pub_fresh() {
  N=$((N + 1))
  D=$T/pub$N
  G=$D/gitea.git
  H=$D/github.git
  W=$D/work
  mkdir -p "$D"
  git init -q --bare -b main "$G"
  git init -q --bare -b main "$H"
  git init -q -b main "$W"
  git -C "$W" remote add origin "$G"
  commit README.md "clean start"
  git -C "$W" push -q origin main
}

start_fake() { # <mode> [password]  -> sets API_URL, API_LOG
  local mode=$1 pw=${2:-$PUB_PASSWORD} i
  API_LOG=$D/api.log
  : >"$API_LOG"
  rm -f "$D/port"
  python3 "$here/fake-gitea.py" --port-file "$D/port" --log "$API_LOG" --user admin --password "$pw" \
    --repo owner/repo --gitea-git "$G" --github-git "$H" --mode "$mode" &
  fake_pids+=("$!")
  FAKE_PID=$!
  for i in $(seq 1 50); do [ -s "$D/port" ] && break; sleep 0.1; done
  API_URL=http://127.0.0.1:$(cat "$D/port")
}
stop_fake() { kill "$FAKE_PID" 2>/dev/null; wait "$FAKE_PID" 2>/dev/null; }

PUB_PASSWORD='p@ss"word\x'"'"'y'
TOKEN_SENTINEL=SENTINEL-TOKEN-0123456789

# Wrapper programs in $T/bin log their arguments, then run the real program.
cat >"$T/bin/curl" <<EOF
#!/bin/sh
printf '%s\n' "\$@" >>"$T/argv-curl.log"
exec "$real_curl" "\$@"
EOF
cat >"$T/bin/git" <<EOF
#!/bin/sh
printf '%s\n' "\$@" >>"$T/argv-git.log"
exec "$real_git" "\$@"
EOF
chmod +x "$T/bin/curl" "$T/bin/git"

run_pub() { # <publish.sh arguments>; the environment comes from PENV
  REPORT=$D/report.txt
  rm -f "$REPORT"
  ( cd "$T" && env PATH="$T/bin:$PATH" DMK_REPO="$W" DMK_REPORT="$REPORT" GITEA_GIT_URL="$G" GITHUB_URL="$H" \
      DMK_POLL_SECONDS=1 "${PENV[@]}" "$P/publish.sh" "$@" ) >/dev/null 2>&1
  PUB_RC=$?
}

expect_pub() { # <name> <exit code> <pattern in the report>
  if [ "$PUB_RC" -ne "$2" ]; then bad "$1" "exit $PUB_RC, expected $2: $(tail -n 3 "$REPORT" 2>/dev/null | tr '\n' ' ')"
  elif [ -n "${3:-}" ] && ! grep -Eq -- "$3" "$REPORT"; then bad "$1" "report lacks /$3/"
  else ok "$1"; fi
}

gh_state() { git -C "$H" for-each-ref --format='%(refname) %(objectname)'; }
gt_state() { git -C "$G" for-each-ref --format='%(refname) %(objectname)' refs/heads refs/tags; }

pub_fresh
PENV=()
run_pub --dry-run
expect_pub "dry run: a clean repository passes (output hidden, exit code and report)" 0 'scans passed. Dry run'
[ -z "$(gh_state)" ] && ok "dry run pushes nothing to GitHub" || bad "dry run pushes nothing to GitHub"

pub_fresh
git -C "$W" checkout -q -b feature
commit f.md "branch with $(ip4 192 168 9 9)"
git -C "$W" push -q --no-verify origin feature
git -C "$W" checkout -q main
run_pub --dry-run
expect_pub "dry run: a finding on a non-default branch blocks (all refs)" 1 'FINDINGS.*file contents'

pub_fresh
W2=$D/work2
git clone -q "$G" "$W2"
git -C "$W2" checkout -q -b other
printf 'x %s\n' "$(ip4 10 8 8 8)" >"$W2/o.md"
git -C "$W2" add -A
git -C "$W2" commit -q -m other
git -C "$W2" push -q --no-verify origin other
run_pub --dry-run
expect_pub "dry run: a finding on a Gitea branch that this clone lacks blocks" 1 'dmk-private-ipv4'

pub_fresh
git -C "$W" checkout -q --detach
commit t.md "tag only $(ip4 172 20 1 1)"
git -C "$W" tag -a only-tag -m "tag"
git -C "$W" push -q --no-verify origin only-tag
git -C "$W" checkout -q main
git -C "$W" branch -q -D feature 2>/dev/null
run_pub --dry-run
expect_pub "dry run: a finding reachable only from a tag blocks" 1 'dmk-private-ipv4'

pub_fresh
head -c 6291456 /dev/zero >"$W/big.dat"
git -C "$W" add -A; git -C "$W" commit -q -m "big"
git -C "$W" rm -q big.dat; git -C "$W" commit -q -m "remove big"
git -C "$W" push -q --no-verify origin main
run_pub --dry-run
expect_pub "dry run: a large file deleted by a later commit still blocks" 1 'BLOCKED size: big.dat'

pub_fresh
commit m.md "a note from $(secret_name)" "message names $(secret_name)"
git -C "$W" push -q --no-verify origin main
run_pub --dry-run
expect_pub "dry run: a private name in a commit message blocks" 1 '\(message\)'

pub_fresh
git -C "$W" tag -a note-tag -m "tag text names $(secret_name)"
git -C "$W" push -q --no-verify origin note-tag
run_pub --dry-run
expect_pub "dry run: a private name in a tag message blocks" 1 '@@ tag'

pub_fresh
git -C "$W" checkout -q -b $(secret_name)-branch
git -C "$W" push -q --no-verify origin $(secret_name)-branch
git -C "$W" checkout -q main
run_pub --dry-run
expect_pub "dry run: a private name in a branch name blocks" 1 '@@ ref'

# upstream base in publish.sh
pub_fresh
commit docs/up.md "upstream $(ip4 192 168 4 4)" "upstream docs"
git -C "$W" tag up-base
commit mine.md "mine, clean" "project note"
git -C "$W" push -q --no-verify origin main
PENV=(DMK_UPSTREAM_BASE=up-base)
run_pub --dry-run
expect_pub "dry run: upstream history is left out when DMK_UPSTREAM_BASE is set" 0 'scans passed'
PENV=()
run_pub --dry-run
expect_pub "dry run: the same history blocks without DMK_UPSTREAM_BASE" 1 'dmk-private-ipv4'

# setup errors
pub_fresh
PENV=(DMK_PRIVATE_RULES="$T/does-not-exist.toml")
run_pub --dry-run
expect_pub "publish refuses without the private rules file" 2 'private rules file is missing'
PENV=()
run_pub --direct-push
expect_pub "publish --direct-push refuses without GH_TOKEN" 2 'GH_TOKEN is not set'
PENV=(GH_TOKEN="$TOKEN_SENTINEL" GITHUB_URL="https://user:secret@github.example.invalid/o/r.git")
run_pub --direct-push
expect_pub "publish refuses a GitHub URL that holds credentials" 2 'holds credentials'
PENV=(GITEA_URL=http://127.0.0.1:1 GITEA_REPO=owner/repo GITEA_ADMIN_USER=admin)
run_pub
expect_pub "publish refuses a sync run without the admin password" 2 'GITEA_ADMIN_PASSWORD is not set'
PENV=()
run_pub --dry-run --no-sync
expect_pub "publish refuses two modes at once" 2 ''

section "publish.sh: sync, push and head comparison"

fake_env() { PENV=(GITEA_URL="$API_URL" GITEA_REPO=owner/repo GITEA_ADMIN_USER=admin GITEA_ADMIN_PASSWORD="$PUB_PASSWORD" DMK_SYNC_TIMEOUT=20 DMK_VERIFY_TIMEOUT=10 "$@"); }

# full success
pub_fresh
git -C "$W" tag -a v1 -m "release one"
git -C "$W" push -q origin v1
start_fake ok
fake_env
: >"$T/argv-curl.log"
run_pub
expect_pub "sync: a clean repository is published and GitHub equals Gitea" 0 'published. GitHub heads and tags equal Gitea'
[ "$(gh_state)" = "$(gt_state)" ] && [ -n "$(gh_state)" ] && ok "sync: GitHub heads and tags equal Gitea after the run" || bad "sync: GitHub heads and tags equal Gitea after the run"
[ "$(sed -n '1p;2p;3p' "$API_LOG" | cut -d' ' -f1 | tr '\n' ' ')" = "GET POST GET " ] && ok "sync: API order is GET before, POST sync, GET after" || bad "sync: API order is GET before, POST sync, GET after" "$(tr '\n' ';' <"$API_LOG")"
grep -q 'auth bad' "$API_LOG" && bad "sync: every API call carried the right login" || ok "sync: every API call carried the right login"
grep -qF -- "$PUB_PASSWORD" "$REPORT" && bad "sync: the report holds no password" || ok "sync: the report holds no password"
grep -qF -- "$PUB_PASSWORD" "$T/argv-curl.log" && bad "sync: the password is in no curl argument" || ok "sync: the password is in no curl argument"
[ "$(stat -c %a "$REPORT")" = 600 ] && ok "sync: the report file has mode 0600" || bad "sync: the report file has mode 0600" "$(stat -c %a "$REPORT")"
stop_fake

# blocked scan: no API call
pub_fresh
commit x.md "bad $(ip4 10 77 77 77)"
git -C "$W" push -q --no-verify origin main
start_fake ok
fake_env
run_pub
expect_pub "sync: a blocked scan exits 1" 1 'Nothing was published'
[ ! -s "$API_LOG" ] && ok "sync: a blocked scan makes no API call" || bad "sync: a blocked scan makes no API call" "$(tr '\n' ';' <"$API_LOG")"
[ -z "$(gh_state)" ] && ok "sync: a blocked scan leaves GitHub empty" || bad "sync: a blocked scan leaves GitHub empty"
stop_fake

# wrong password
pub_fresh
start_fake ok 'another-password'
fake_env
run_pub
expect_pub "sync: a wrong admin password fails with exit 3" 3 'cannot read the push mirrors|HTTP 401'
grep -qF -- "$PUB_PASSWORD" "$REPORT" && bad "sync: a failed login does not put the password in the report" || ok "sync: a failed login does not put the password in the report"
stop_fake

# error mode
pub_fresh
start_fake error
fake_env
run_pub
expect_pub "sync: a mirror that reports last_error fails with exit 3" 3 'mirror reported an error'
stop_fake

# nomove
pub_fresh
start_fake nomove
fake_env DMK_SYNC_TIMEOUT=3
run_pub
expect_pub "sync: a mirror whose last_update never moves fails with exit 3" 3 'last_update did not move'
stop_fake

# nomirror
pub_fresh
start_fake nomirror
fake_env
run_pub
expect_pub "sync: a repository without a push mirror fails with exit 3" 3 'no push mirror'
stop_fake

# heads differ after a sync that pushed nothing
pub_fresh
start_fake nopush
fake_env DMK_VERIFY_TIMEOUT=3
run_pub
expect_pub "sync: GitHub heads that differ from Gitea fail with exit 4" 4 'refs that differ'
stop_fake

# --no-sync
pub_fresh
git -C "$G" push -q --prune "$H" '+refs/heads/*:refs/heads/*' '+refs/tags/*:refs/tags/*'
PENV=(DMK_VERIFY_TIMEOUT=3)
run_pub --no-sync
expect_pub "no-sync: equal heads pass" 0 'step c: skipped'
git -C "$H" update-ref -d refs/heads/main
run_pub --no-sync
expect_pub "no-sync: a missing GitHub head fails with exit 4" 4 'refs that differ'

# --direct-push
pub_fresh
git -C "$W" tag -a v9 -m "nine"
git -C "$W" checkout -q -b topic
commit t.md "topic, clean"
git -C "$W" push -q origin main topic v9
git -C "$W" checkout -q main
git -C "$H" fetch -q "$G" 'refs/heads/main:refs/heads/stale-only-on-github'
: >"$T/argv-git.log"
PENV=(GH_TOKEN="$TOKEN_SENTINEL" DMK_VERIFY_TIMEOUT=5)
run_pub --direct-push
expect_pub "direct push: the scans pass and GitHub equals Gitea" 0 'published. GitHub heads and tags equal Gitea'
[ "$(gh_state)" = "$(gt_state)" ] && ok "direct push: GitHub holds the Gitea heads and tags" || bad "direct push: GitHub holds the Gitea heads and tags" "$(gh_state | tr '\n' ';')"
git -C "$H" rev-parse -q --verify refs/heads/stale-only-on-github >/dev/null && bad "direct push: --prune removes a GitHub-only branch" || ok "direct push: --prune removes a GitHub-only branch"
grep -qF -- "$TOKEN_SENTINEL" "$REPORT" && bad "direct push: the report holds no token" || ok "direct push: the report holds no token"
grep -qF -- "$TOKEN_SENTINEL" "$T/argv-git.log" && bad "direct push: the token is in no git argument" || ok "direct push: the token is in no git argument"

# a rewritten Gitea history reaches GitHub (force)
git -C "$W" checkout -q topic
git -C "$W" reset -q --hard HEAD~1
commit t2.md "topic rewritten"
git -C "$W" push -q -f origin topic
git -C "$W" checkout -q main
run_pub --direct-push
expect_pub "direct push: a rewritten Gitea branch is forced to GitHub" 0 'published'
[ "$(gh_state)" = "$(gt_state)" ] && ok "direct push: GitHub follows the rewritten branch" || bad "direct push: GitHub follows the rewritten branch"

# direct push with a finding does not push
pub_fresh
commit bad.md "x $(ip4 192 168 50 50)"
git -C "$W" push -q --no-verify origin main
PENV=(GH_TOKEN="$TOKEN_SENTINEL")
run_pub --direct-push
expect_pub "direct push: a blocked scan exits 1" 1 'Nothing was published'
[ -z "$(gh_state)" ] && ok "direct push: a blocked scan pushes nothing" || bad "direct push: a blocked scan pushes nothing"

# the credential helper
section "publish.sh: the GitHub credential helper"
(
  . "$P/lib.sh"
  dmk_write_gh_credential_helper "$T/helper.sh"
)
got=$(printf 'protocol=https\nhost=github.com\n\n' | GH_TOKEN="$TOKEN_SENTINEL" "$real_git" -c credential.helper= -c "credential.helper=!$T/helper.sh" credential fill 2>/dev/null)
printf '%s\n' "$got" | grep -qx "password=$TOKEN_SENTINEL" && printf '%s\n' "$got" | grep -qx 'username=x-access-token' && ok "the helper answers git with the token from the environment" || bad "the helper answers git with the token from the environment"
grep -qF -- "$TOKEN_SENTINEL" "$T/helper.sh" && bad "the helper file does not hold the token" || ok "the helper file does not hold the token"

# ================================================================ the real private rules file

section "the real private rules file (skipped when it does not exist)"
real_rules=$(getent passwd "$(id -un)" | cut -d: -f6)/.config/decision-model-kernels/gitleaks-private.toml
if [ -f "$real_rules" ]; then
  mkdir -p "$T/real-ok" "$T/real-bad"
  printf 'Source: https://github.com/WyrmKeep/decision-model-kernels\n' >"$T/real-ok/a.md"
  cat >"$T/real-ok/LICENSE" <<'EOF'
MIT License

Copyright (c) 2026 WyrmKeep
EOF
  cat >"$T/real-ok/english.md" <<'EOF'
The kernel launches one thread block for each tile. Latency is the time between a
request and its first token. A quantized weight uses four bits. The model is
loaded once, and the server keeps it in memory. Version 12.8.1 of the toolkit
adds nothing here. Listen on port 8080 at localhost. See docs/build.md for more.
EOF
  printf 'owner %s\n' "$(owner_name)" >"$T/real-bad/a.md"
  printf 'see https://github.com/%s/x\n' "$(owner_name)" >"$T/real-bad/b.md"
  ( DMK_PRIVATE_RULES="$real_rules" "$P/scan-dir.sh" "$T/real-ok" >/dev/null 2>&1 ) && ok "real rules: the allowed URL, the license line and normal English pass" || bad "real rules: the allowed URL, the license line and normal English pass"
  ( DMK_PRIVATE_RULES="$real_rules" "$P/scan-dir.sh" "$T/real-bad" >"$OUT" 2>&1 ) && bad "real rules: the owner name outside the allowed forms blocks" || { grep -q 'a.md:1 dmk-private-owner-name' "$OUT" && grep -q 'b.md:1 dmk-private-owner-name' "$OUT" && ok "real rules: the owner name outside the allowed forms blocks" || bad "real rules: the owner name outside the allowed forms blocks"; }
  [ "$(stat -c %a "$real_rules")" = 600 ] && ok "real rules: the file has mode 0600" || bad "real rules: the file has mode 0600"
else
  printf 'SKIP  no real private rules file on this machine\n'
fi

# ================================================================ summary

printf '\n%s passed, %s failed\n' "$pass" "$fail"
if [ "$fail" -ne 0 ]; then
  printf 'failed tests:\n'
  printf '  %s\n' "${fails[@]}"
  exit 1
fi
exit 0
