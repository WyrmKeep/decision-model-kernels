# Publication gate

The publication gate keeps private data out of the public repositories. It scans every push to Gitea. It scans all refs again before anything goes to GitHub.

## What the gate does

Gitea is the place of work. GitHub holds the public copy. Nothing reaches GitHub except through this gate.

The gate has two parts:

| Part | When it runs | What it does |
|---|---|---|
| `pre-push` hook | At each `git push` | Scans only the new commits. Refuses the push on any finding. |
| `publish.sh` | When you run it, at the end of a phase | Scans all refs. Starts the mirror sync only if the scan is clean. Then checks that GitHub equals Gitea. |

## Why CI is never the privacy check

A scan in CI runs after the push. At that time the data is already on the server. The push mirror can send it to GitHub in a few seconds. A leak that CI finds is already public. CI can report a problem, but it cannot stop it. The gate runs on your machine, before the data leaves it.

## What the gate blocks

| Check | What it finds | Rule source |
|---|---|---|
| Secrets | Tokens, keys and passwords | The default gitleaks rules |
| Private addresses | Private IPv4 ranges (RFC 1918), the carrier-grade NAT range, the Tailscale IPv6 prefix | `gitleaks.toml` |
| Tailnet names | Host names that end in the Tailscale domain | `gitleaks.toml` |
| Identifiers | UUIDs | `gitleaks.toml` |
| Server paths | Mount points of a storage server, home directories, the home directory of the root user | `gitleaks.toml` |
| Monitor numbers | Numbers of uptime monitors | `gitleaks.toml` |
| Internal names | Machine names, domains, user names, project code names, vault item IDs, the owner account name | The private rules file |
| Large files | Files above the size limit (default 5 MiB) | `lib.sh` |
| File types | Model files, binary files, reports and archives | `lib.sh` |

The three documentation networks of RFC 5737 pass. These are 192.0.2.0/24, 198.51.100.0/24 and 203.0.113.0/24.

The gate lets two forms of the owner account name through. One form is a `github.com/WyrmKeep/` URL. The other form is this exact line:

```
Copyright (c) 2026 WyrmKeep
```

The blocked file types are `*.gguf`, `*.safetensors`, `*.bin`, `*.pt`, `*.onnx`, `*.nsys-rep`, `*.ncu-rep`, `*.sqlite`, `*.sqlite3`, `*.db`, `*.pdf`, `*.tar`, `*.tar.*`, `*.tgz`, `*.zip` and `*.7z`.

## What the scans read

Each scan reads more than file contents. It reads this text:

- The added lines of each commit.
- Commit messages.
- Author and committer names and addresses.
- Branch names and tag names.
- Tag messages.
- The lines that a merge commit adds in its conflict resolution.

The size and type check reads history, not only the current trees. A large file that a later commit deletes is still public once the history is public.

Each text scan runs gitleaks twice:

1. The first pass uses the default rules, the public rules and the private rules.
2. The second pass uses the public rules and the private rules alone.

The default rules skip some files, for example SVG images. The second pass closes that gap. The gate also ignores `gitleaks:allow` comments and any `.gitleaksignore` file in the repository.

## Requirements

- bash 4.4 or later
- git
- gitleaks 8.30 or later (the tests use 8.30.1)
- jq and curl, for the mirror sync in `publish.sh`
- python3, for the tests only

## Install

Do these steps once on each clone.

1. Make sure that `gitleaks version` works.
2. Make the private rules file. See "The private rules file" below.
3. Run `tools/publish/install-hooks.sh <clone>`.
4. Add the upstream base to the llama.cpp fork: `tools/publish/install-hooks.sh --upstream-base "<revisions>" <clone>`.
5. Run `tools/publish/install-hooks.sh --verify <clone>`.

The script copies the hook to `.git/hooks/pre-push`. It copies the support files to `.git/hooks/dmk-publish/`. It records the SHA-256 checksum of every file in `dmk-publish/SHA256SUMS`. The hook checks the checksums at each push.

Run `install-hooks.sh` again after each change to `lib.sh`, `gitleaks.toml` or `allow-files.txt`. The script refuses to replace a hook that is not the gate hook. Use `--force` to move that hook aside. The script refuses to run when `core.hooksPath` is set.

## The upstream base

The llama.cpp fork contains the history of upstream llama.cpp. That history has addresses and paths in its own documents. The project rules must not scan it.

Set the upstream base to the revisions that hold upstream history. Use a space between revisions. Example: `b12004 upstream/master`. The gate leaves out every commit that these revisions reach.

- Set the variable `DMK_UPSTREAM_BASE`, or
- set the clone setting `dmk.upstreamBase` (the `--upstream-base` option does this).

Leave the base empty in the project repository. When you merge newer upstream commits, add the new upstream revision to the base. If a revision does not resolve, the gate refuses to run.

## The private rules file

The private rules file lists the internal names that must stay private. The file is never part of a repository.

The default path is `~/.config/decision-model-kernels/gitleaks-private.toml`. Set `DMK_PRIVATE_RULES` to use another path. The mode must be 0600. The hook refuses to push when the file is missing.

1. Make a copy of `tools/publish/private-rules.example.toml` at the default path.
2. Set the mode: `chmod 600 <file>`.
3. Replace each example name with a real internal name.
4. Add one rule for each kind of name you must hide.
5. Do a test of each rule on normal English text.

The file holds `[[rules]]` blocks only. Do not add an `[extend]` section or top-level keys. The gate adds them.

Cover these kinds of name:

- Machine names and aliases.
- The tailnet domain and other private domains.
- Host and port pairs of internal services.
- User names and personal names.
- Code names of other private projects.
- Vault item IDs.
- The owner account name. Keep the two allowed forms from the example.

A rule that fires on ordinary words is noise. Use word boundaries for short names. Keep the file out of every repository. If it is ever committed, the gate blocks the push, because the rules match their own text.

## Settings

| Variable | Meaning | Default |
|---|---|---|
| `DMK_PRIVATE_RULES` | Path of the private rules file | `~/.config/decision-model-kernels/gitleaks-private.toml` |
| `DMK_UPSTREAM_BASE` | Revisions of upstream history that no scan reads | Clone setting `dmk.upstreamBase`, then none |
| `DMK_MAX_FILE_BYTES` | File size limit in bytes | 5242880 |
| `DMK_ALLOWLIST` | Path of the file allow-list | The copy installed with the hook |
| `DMK_IGNORE_FILE` | Path of a gitleaks fingerprint ignore file | None |
| `DMK_GITLEAKS` | Path of the gitleaks program | `gitleaks` from `PATH` |
| `DMK_REPO` | Clone that `publish.sh` publishes | The clone that holds the script |
| `DMK_REPORT` | Path of the report file of `publish.sh` | `<git dir>/dmk-publish-report.txt` |
| `GITEA_URL` | Base URL of Gitea, for the API | None |
| `GITEA_REPO` | `owner/name` of the repository in Gitea | None |
| `GITEA_ADMIN_USER` | Name of the Gitea admin user | None |
| `GITEA_ADMIN_PASSWORD` | Gitea admin password | None |
| `GITEA_GIT_URL` | Git URL of the Gitea repository | URL of the remote `origin` |
| `GITHUB_URL` | Git URL of the GitHub repository, without credentials | None |
| `GH_TOKEN` | GitHub token, for `--direct-push` only | None |
| `DMK_SYNC_TIMEOUT` | Seconds to wait for the mirror sync | 180 |
| `DMK_VERIFY_TIMEOUT` | Seconds to wait for equal heads | 60 |

The file allow-list is `allow-files.txt`. It lists one path or glob in each line. A listed path passes the size and type check. Write the reason in a comment. Run `install-hooks.sh` again after a change.

## Publish

Publication is a step that you take on purpose. Do it at the end of a phase or a pull request. A push never publishes.

1. Push all work to Gitea. The hook scans each push.
2. Run a dry run: `tools/publish/publish.sh --dry-run`.
3. Read the report. The path is in `DMK_REPORT`.
4. Set `GITEA_URL`, `GITEA_REPO`, `GITEA_ADMIN_USER`, `GITEA_ADMIN_PASSWORD` and `GITHUB_URL`.
5. Run `tools/publish/publish.sh`.
6. Read the exit code and the report.

Do not write the password in a file of the repository. Run the script under a credential helper that sets the variables for one run. The helper can hide standard output and standard error. The script writes a report without secrets and sets the exit code, so it works under such a helper.

The script runs these steps:

| Step | Action |
|---|---|
| Setup | Fetches all refs of Gitea into `refs/dmk-gitea/`. The mirror sync copies the data that Gitea holds. So the scan reads that data. |
| a | Runs gitleaks over all refs, without the upstream base. |
| b | Runs the size and type check over the history of all refs. |
| c | Only if a and b pass: sends `POST push_mirrors-sync` to Gitea. Then reads `push_mirrors` until `last_update` moves and `last_error` is empty. |
| d | Compares `git ls-remote` of Gitea and GitHub. Every head and tag must be equal. |

If step a or b finds anything, the script makes no API call and no push.

### Options

| Option | Effect |
|---|---|
| `--dry-run` | Runs steps a and b only. |
| `--no-sync` | Skips step c. Runs the scans and step d. |
| `--direct-push` | Replaces step c with a `git push` to GitHub. |

Use `--direct-push` only when Gitea cannot turn off the periodic mirror sync. The script pushes the Gitea heads and tags with `--prune`. It forces the update, so GitHub always equals Gitea. It reads the token from `GH_TOKEN` through a temporary credential helper. The token is never in the URL or in an argument.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | The scans passed. The run is complete. |
| 1 | A scan found something. The gate published nothing. |
| 2 | Setup error: a missing tool, variable or file. |
| 3 | The mirror sync failed. |
| 4 | GitHub differs from Gitea. |
| 5 | The direct push failed. |

## What to do when the gate finds something

The output shows the rule name and the place of each finding: file and line, or commit and line. It never shows the text. Open the place and look at the text yourself.

### A hook refuses a push

The refused push sent nothing to Gitea.

1. Read the findings.
2. Remove the item from the files, or change the text.
3. For a message or identity, amend the commit or rewrite the history.
4. Push again.

If the item was a real secret, check where else it has been. Rotate it if anyone else could read it.

### A finding is a false positive

1. Change the text if you can.
2. For a file, add its path to `allow-files.txt`. Write the reason.
3. For one finding, put its fingerprint in a file and set `DMK_IGNORE_FILE`. The fingerprint is `commit:file:rule:line`.
4. For a rule that fires on ordinary words, make the rule narrower.
5. Run `install-hooks.sh` again.

### The scan of `publish.sh` finds a secret on Gitea

Gitea is not public, but the secret is still in Gitea. GitHub has no copy yet.

1. Rotate the secret.
2. Remove it from the history of every ref in a clone.
3. Force-push the rewritten refs to Gitea.
4. Delete the old branches and tags that hold it.
5. Run the garbage collection of Gitea, so it drops the old objects.
6. Run `publish.sh` again.

### A secret reached GitHub

Treat the secret as public. Rotate it first. A rewrite of history cannot recall copies that other people made.

1. Rotate the secret.
2. Remove it from history as described above.
3. Publish the rewritten history through the gate.
4. Ask GitHub support to remove cached views and pull request refs of the old commits.

### The sync fails or GitHub differs

1. Read the report. It shows the HTTP status or the sync error.
2. Examine the push mirror in Gitea. Make sure that the mirror credentials are correct.
3. Run `publish.sh --no-sync` to compare the heads again.

## Tests

Run `tools/publish/test/run-tests.sh`. The tests build repositories in `/tmp` and use the example file with invented names. Local bare repositories stand in for Gitea and GitHub. A fake HTTP server stands in for the Gitea API. The tests never use a network host.

The tests prove each case:

| Case | Result |
|---|---|
| A private IPv4 address (each private range and the carrier-grade NAT range) | Blocked |
| A tailnet host name | Blocked |
| A name from the private rules file | Blocked |
| A UUID, a server path or a monitor number | Blocked |
| A file of 6 MiB | Blocked |
| A `.gguf` file and each other blocked type | Blocked |
| A clean commit | Passes |
| The documentation networks | Pass |
| The allowed owner URL and the license line | Pass |
| An address in upstream history, with the upstream base set | Skipped |
| The same history, without the upstream base | Blocked |
| No private rules file | Push refused |
| `publish.sh` with a finding | Exit 1, no API call |
| `publish.sh` sync and head comparison | Each success and each failure |

## Files

| File | Purpose |
|---|---|
| `pre-push` | The git hook |
| `install-hooks.sh` | Installs the hook and records checksums |
| `publish.sh` | The publication gate |
| `scan-dir.sh` | Scans a directory with the gate rules |
| `lib.sh` | Shared functions |
| `gitleaks.toml` | The public rules. It extends the default rules. |
| `private-rules.example.toml` | Example of the private rules file |
| `allow-files.txt` | The file allow-list |
| `test/run-tests.sh` | The tests |
| `test/fake-gitea.py` | The fake Gitea API for the tests |

## Limits

- The gate reads text. It does not read text inside images.
- The gate finds only the names that the rules list. The private rules file does not list every name, so an unlisted name passes.
- The hook runs on your machine. The option `--no-verify` skips it. That is why `publish.sh` scans again.
- A new clone has no hook. Install it before the first push.
- `scan-dir.sh` refuses to run when the directory has a `.gitleaksignore` file.
