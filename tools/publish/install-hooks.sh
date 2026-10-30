#!/usr/bin/env bash
# Install the publication gate pre-push hook into a clone.
#
# Usage: install-hooks.sh [--force] [--upstream-base <revisions>] <clone>
#        install-hooks.sh --verify <clone>
#
# The script copies the hook to <clone>/.git/hooks/pre-push and the support
# files (lib.sh, gitleaks.toml, allow-files.txt) to <clone>/.git/hooks/dmk-publish/.
# It records the SHA-256 checksum of every copied file in dmk-publish/SHA256SUMS.
# The hook checks those checksums at the start of each push.
# --upstream-base stores the revisions of an upstream project (for the llama.cpp
# fork) in the clone setting dmk.upstreamBase. The hook never scans that history.
set -u
export LC_ALL=C

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
force=0
verify=0
base=
clone=

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case $1 in
    --force) force=1 ;;
    --verify) verify=1 ;;
    --upstream-base)
      shift
      [ $# -gt 0 ] || { echo "install-hooks: --upstream-base needs a value" >&2; exit 2; }
      base=$1
      ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "install-hooks: unknown option $1" >&2; exit 2 ;;
    *) [ -z "$clone" ] || { echo "install-hooks: one clone only" >&2; exit 2; }; clone=$1 ;;
  esac
  shift
done
[ -n "$clone" ] || { usage >&2; exit 2; }

git -C "$clone" rev-parse --git-dir >/dev/null 2>&1 || { echo "install-hooks: $clone is not a git repository" >&2; exit 2; }
hooks=$(git -C "$clone" rev-parse --path-format=absolute --git-path hooks) || exit 2
support=$hooks/dmk-publish

if [ "$verify" -eq 1 ]; then
  [ -f "$support/SHA256SUMS" ] || { echo "install-hooks: no gate installed in $hooks" >&2; exit 1; }
  if (cd "$hooks" && sha256sum -c dmk-publish/SHA256SUMS); then
    echo "install-hooks: installed gate files match their recorded checksums."
    exit 0
  fi
  echo "install-hooks: installed gate files differ from the recorded checksums." >&2
  exit 1
fi

hp=$(git -C "$clone" config --get core.hooksPath || true)
if [ -n "$hp" ]; then
  echo "install-hooks: core.hooksPath is set ($hp). Unset it or install the hook there by hand." >&2
  exit 2
fi

for f in pre-push lib.sh gitleaks.toml; do
  [ -f "$here/$f" ] || { echo "install-hooks: $here/$f is missing" >&2; exit 2; }
done
allow=$here/allow-files.txt
[ -f "$allow" ] || { echo "install-hooks: $allow is missing" >&2; exit 2; }

if [ -e "$hooks/pre-push" ] && ! grep -q 'DMK-PUBLICATION-GATE' "$hooks/pre-push"; then
  if [ "$force" -ne 1 ]; then
    echo "install-hooks: $hooks/pre-push exists and is not the gate hook. Use --force to move it aside." >&2
    exit 2
  fi
  mv -- "$hooks/pre-push" "$hooks/pre-push.before-dmk-gate"
fi

mkdir -p "$support"
install -m 0755 "$here/pre-push" "$hooks/pre-push"
install -m 0644 "$here/lib.sh" "$support/lib.sh"
install -m 0644 "$here/gitleaks.toml" "$support/gitleaks.toml"
install -m 0644 "$allow" "$support/allow-files.txt"
(cd "$hooks" && sha256sum pre-push dmk-publish/lib.sh dmk-publish/gitleaks.toml dmk-publish/allow-files.txt >dmk-publish/SHA256SUMS)

if [ -n "$base" ]; then
  git -C "$clone" config --local dmk.upstreamBase "$base"
fi

echo "install-hooks: installed in $hooks"
echo "install-hooks: SHA-256 of the hook: $(cut -d' ' -f1 < <(grep ' pre-push$' "$support/SHA256SUMS"))"
echo "install-hooks: checksums recorded in $support/SHA256SUMS"
if [ ! -f "${DMK_PRIVATE_RULES:-$HOME/.config/decision-model-kernels/gitleaks-private.toml}" ]; then
  echo "install-hooks: WARNING the private rules file does not exist yet. Pushes are refused until it does." >&2
fi
exit 0
