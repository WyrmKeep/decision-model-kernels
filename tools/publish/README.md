# Publication gate

The publication gate stops private data before it goes to the public repositories. It scans each git push to Gitea. It scans all refs again before data goes to GitHub.

## Tasks of the publication gate

The work of the project is in Gitea. GitHub holds the public repository. Data goes to GitHub only through the publication gate.

The publication gate has two parts:

| Part | When it runs | Task |
|---|---|---|
| `pre-push` hook | At each `git push` | Scans only the new commits. Stops the git push if it finds an item. |
| `publish.sh` | When you run it at the end of a part of the work | Scans all refs. Starts the mirror sync only if the scan finds no item. Then makes sure that GitHub has the same branches and tags as Gitea. |

## CI is not the check for private data

A CI scan runs after the git push. At that time the data is on the server. The push mirror can send the data to GitHub in a small number of seconds. When CI finds private data, each person can read the data on GitHub. CI can report a problem, but it cannot stop the problem. The publication gate runs on your device, before the data goes to a different device.

## Data that the publication gate stops

| Check | Data that it finds | Rules in |
|---|---|---|
| Secrets | Access tokens, access keys and other secrets | The default gitleaks rules |
| IP addresses of local networks | The IPv4 ranges of RFC 1918, the carrier-grade NAT range and the IP addresses of Tailscale | `gitleaks.toml` |
| Tailscale names | Host names with the domain name of the Tailscale network | `gitleaks.toml` |
| Codes of items | Each code of the type UUID | `gitleaks.toml` |
| Server paths | Paths of the disks of a server, paths of the folders where persons keep their files, and the path of the folder of the user `root` | `gitleaks.toml` |
| Numbers after `monitor` or `kuma` | Text with `monitor` or `kuma`, the mark `#` and a number | `gitleaks.toml` |
| Internal names | Host names, domain names, user names, code names of projects, vault item IDs and the name of the owner account | The private rules file |
| Large files | Files that are larger than the limit (default 5 MiB) | `lib.sh` |
| File types | Model files, binary files, reports and archives | `lib.sh` |

The file check makes the last two checks. gitleaks makes the other checks.

The publication gate lets the three example networks of RFC 5737 through. The networks have the ranges 192.0.2.0/24, 198.51.100.0/24 and 203.0.113.0/24.

The publication gate lets the name of the owner account through in two shapes. One shape is a `github.com/WyrmKeep/` URL. The other shape is a line with only this text:

```
Copyright (c) 2026 WyrmKeep
```

The publication gate stops these file types: `*.gguf`, `*.safetensors`, `*.bin`, `*.pt`, `*.onnx`, `*.nsys-rep`, `*.ncu-rep`, `*.sqlite`, `*.sqlite3`, `*.db`, `*.pdf`, `*.tar`, `*.tar.*`, `*.tgz`, `*.zip` and `*.7z`.

## Text that the scans read

Each scan reads more than the contents of files. It reads this text:

- The lines that each commit adds.
- Commit messages.
- The names and the email addresses of the person who writes a commit and of the person who commits it.
- Branch names and tag names.
- Tag messages.
- The lines that a merge commit adds for a merge conflict.

The file check reads the history, not only the files that the branches and tags hold at this time. If a subsequent commit removes a large file, the file stays in the history. When the history goes to GitHub, each person can read the file.

Each text scan runs gitleaks two times:

1. The first run uses the default gitleaks rules, the gitleaks rules in `gitleaks.toml` and the gitleaks rules of the private rules file.
2. The second run uses only the gitleaks rules in `gitleaks.toml` and the gitleaks rules of the private rules file.

The default gitleaks rules do not read some files, for example `.svg` files. The second run reads those files. The publication gate also ignores the text `gitleaks:allow` and each `.gitleaksignore` file in the repository.

## Necessary tools

- `bash` 4.4 or higher
- git
- gitleaks 8.30 or higher (the tests use 8.30.1)
- `jq` and `curl`, for the mirror sync in `publish.sh`
- `python3`, for the tests only

## Install

Do these steps one time in each clone.

1. Make sure that `gitleaks version` works.
2. Make the private rules file. Read the section "The private rules file" in this document.
3. Run `tools/publish/install-hooks.sh <clone>`.
4. Add the upstream base to the llama.cpp fork with `tools/publish/install-hooks.sh --upstream-base "<revisions>" <clone>`.
5. Run `tools/publish/install-hooks.sh --verify <clone>`.

The script copies the hook to `.git/hooks/pre-push`. It copies the files `lib.sh`, `gitleaks.toml` and `allow-files.txt` to `.git/hooks/dmk-publish/`. It records the SHA-256 checksum of each file in `dmk-publish/SHA256SUMS`. At each git push, the hook compares the files with the checksums.

Run `install-hooks.sh` again after each change to `lib.sh`, `gitleaks.toml` or `allow-files.txt`. The script does not replace a hook that is not the hook of the publication gate. Use `--force` to move that hook to a different name. The script does not install if `core.hooksPath` is set. The flag `--verify` does not do this test.

## The upstream base

The llama.cpp fork holds the history of upstream llama.cpp. That history has IP addresses and paths in its documents. The gitleaks rules of the project must not scan it.

Set the upstream base to the revisions that hold the upstream history. Put a space between the revisions. Example: `b12004 upstream/master`. The publication gate does not scan the history of these revisions.

- Set the environment variable `DMK_UPSTREAM_BASE`, or
- set the value `dmk.upstreamBase` in the clone with `git config` (the flag `--upstream-base` does this).

Keep the upstream base empty in the project repository. When you merge new upstream commits, add the new upstream revision to the upstream base. If a revision does not name a commit, the publication gate does not run.

## The private rules file

The private rules file has the gitleaks rules for the internal names of the project. Only the persons of the project can know these names. A repository must not hold the file.

The default path is `~/.config/decision-model-kernels/gitleaks-private.toml`. Set `DMK_PRIVATE_RULES` to use a different path. Set the file mode to 0600 or 0400. If the file mode is different, the scripts show a warning and continue. The hook stops the git push if the file is missing.

1. Make a copy of `tools/publish/private-rules.example.toml` at the default path.
2. Set the file mode with `chmod 600 <file>`.
3. Replace each example name with an internal name of the project.
4. Add one gitleaks rule for each type of internal name.
5. Do a test of each gitleaks rule on a text with usual words.

The file holds only entries that start with `[[rules]]`. Do not add `[extend]` or keys at the top of the file. The publication gate adds them.

Add rules for these types of name:

- Host names and the other names of the same hosts.
- The domain name of the Tailscale network and other internal domain names.
- Pairs of a host name and a port of internal servers.
- User names and the names of persons.
- Code names of other internal projects.
- Vault item IDs.
- The name of the owner account. Keep the two shapes that the example lets through.

A gitleaks rule that finds usual words causes noise. For a short name, write the gitleaks rule to find the name only as a word. Do not put the file in a repository. If you commit the file, the publication gate stops the git push, because each gitleaks rule finds the text of the rule.

## Environment variables

| Environment variable | Meaning | Default |
|---|---|---|
| `DMK_PRIVATE_RULES` | Path of the private rules file | `~/.config/decision-model-kernels/gitleaks-private.toml` |
| `DMK_UPSTREAM_BASE` | Revisions of upstream history that no scan reads | The value `dmk.upstreamBase` of the clone, or none if it is not set |
| `DMK_MAX_FILE_BYTES` | The largest file that the file check lets through, in bytes | 5242880 |
| `DMK_ALLOWLIST` | Path of the allow-list | Hook: the copy that `install-hooks.sh` installs with the hook. `publish.sh`: the file `allow-files.txt` in the folder of the script. `scan-dir.sh` does not use an allow-list. |
| `DMK_IGNORE_FILE` | Path of a file with gitleaks fingerprints to ignore | `publish.sh`: the file `gitleaks.ignore` in the folder of the script, if the folder has that file. Hook and `scan-dir.sh`: none. |
| `DMK_GITLEAKS` | Path of the gitleaks program | `gitleaks` from `PATH` |
| `DMK_REPO` | Clone that `publish.sh` publishes | The clone that holds the script |
| `DMK_REPORT` | Path of the report file of `publish.sh` | `<git dir>/dmk-publish-report.txt` |
| `GITEA_URL` | URL of Gitea for the API. Write it without the API path, because the script adds `/api/v1/`. | None |
| `GITEA_REPO` | `owner/name` of the repository in Gitea | None |
| `GITEA_ADMIN_USER` | User name of the admin account of Gitea | None |
| `GITEA_ADMIN_PASSWORD` | Secret of the admin account of Gitea | None |
| `GITEA_GIT_URL` | Git URL of the Gitea repository | URL of the remote `origin` |
| `GITHUB_URL` | Git URL of the GitHub repository, without a user name and a secret | None |
| `GH_TOKEN` | GitHub access token, for `--direct-push` only | None |
| `DMK_SYNC_TIMEOUT` | Seconds to wait for the mirror sync | 180 |
| `DMK_VERIFY_TIMEOUT` | Seconds to wait until GitHub and Gitea have the same refs | 60 |
| `DMK_HTTP_TIMEOUT` | Time limit of one request to the Gitea API, in seconds | 30 |
| `DMK_POLL_SECONDS` | Seconds between two reads of the Gitea API or of the refs, while the script waits | 3 |

The allow-list is the file `allow-files.txt`. Each line has one path. A path can have the mark `*`. The file check lets each path in the allow-list through. Write the cause after the mark `#`. Run `install-hooks.sh` again after a change.

## Publish

Publish at the end of a part of the work or at the end of a pull request. You select the time to publish. A git push does not publish.

1. Push all work to Gitea. The hook scans each git push.
2. Run `tools/publish/publish.sh --dry-run`. The script scans and sends no data.
3. Read the report. `DMK_REPORT` gives its path.
4. Set `GITEA_URL`, `GITEA_REPO`, `GITEA_ADMIN_USER`, `GITEA_ADMIN_PASSWORD` and `GITHUB_URL`.
5. Run `tools/publish/publish.sh`.
6. Read the exit code and the report.

Do not write the value of `GITEA_ADMIN_PASSWORD` in a file of the repository. Run the script with a credential helper that sets the environment variables for one run. The credential helper can remove the standard output and the standard error of the script from your terminal. The script works with such a credential helper, because it writes a report without secrets and sets the exit code.

The script runs these steps:

| Step | Task |
|---|---|
| Start | Gets all refs of Gitea and puts them in `refs/dmk-gitea/`. The mirror sync copies the data that Gitea holds. Because of this, the scan reads that data. |
| `a` | Runs gitleaks on all refs. It scans the file contents and the commit text that the section "Text that the scans read" shows. It does not read the history of the upstream base. |
| `b` | Runs the file check on the history of all refs. |
| `c` | Only if `a` and `b` find no item: sends `POST push_mirrors-sync` to Gitea. Then reads `push_mirrors` until `last_update` changes and `last_error` is empty. |
| `d` | Compares the output of `git ls-remote` for Gitea and for GitHub. Each branch and each tag must be the same. |

If step `a` or `b` finds an item, the script does not use the API and does not push.

### Flags

| Flag | Effect |
|---|---|
| `--dry-run` | Runs steps `a` and `b` only. |
| `--no-sync` | Does not run step `c`. Runs the scans and step `d`. |
| `--direct-push` | Replaces step `c` with a `git push` to GitHub. |

Use `--direct-push` only if Gitea cannot stop the mirror sync that it does at an interval. The script pushes the branches and tags of Gitea with `--prune`. It does a force push. Because of this, GitHub always has the same branches and tags as Gitea. It reads the token from `GH_TOKEN` through a temporary credential helper. The token is not in the URL and is not in the command.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | The scans find no item. The run completes. |
| 1 | A scan finds an item. The publication gate publishes no data. |
| 2 | Setup error: a tool, an environment variable or a file is missing. |
| 3 | The mirror sync gives an error. |
| 4 | GitHub has different branches or tags than Gitea. |
| 5 | The `git push` of `--direct-push` gives an error. |

## If the publication gate finds an item

The output shows the name of the gitleaks rule. It also shows the position of each item: the file and the line, or the commit and the line. It does not show the text. Go to the position and look at the text.

### The hook stops a git push

The hook stops the git push. No data goes to Gitea.

1. Read the items in the output.
2. Remove the item from the files, or change the text.
3. For a commit message, a name or an email address, change the commit or rewrite the history.
4. Push again.

If the item is a secret, examine the other areas where the secret is. If other persons can read it, replace it with a new secret.

### An item is a false positive

1. Change the text if you can.
2. Let a file through: add its path to `allow-files.txt` and write the cause.
3. Ignore one item: put its fingerprint in a file and set `DMK_IGNORE_FILE` to the path of that file. The fingerprint is `commit:file:rule:line`.
4. Change a gitleaks rule that finds usual words. The new gitleaks rule must find less text.
5. Run `install-hooks.sh` again.

### The scan of `publish.sh` finds a secret in Gitea

Only we can read Gitea, but the secret is in Gitea. At this time GitHub has no copy.

1. Replace the secret with a new secret.
2. Remove the secret from the history of all refs in a clone.
3. Do a force push of the refs after the rewrite to Gitea.
4. Remove the branches and tags that hold the secret.
5. Run the garbage collection of Gitea. It removes the objects that no ref holds.
6. Run `publish.sh` again.

### A secret is on GitHub

Think that each person can read the secret. Replace it first. A rewrite of the history cannot remove the copies that other persons made.

1. Replace the secret with a new secret.
2. Remove the secret from the history with the steps for a secret in Gitea.
3. Publish the history after the rewrite through the publication gate.
4. Tell GitHub to remove the views in its cache and the pull request refs of those commits.

### The mirror sync gives an error or GitHub is different

1. Read the report. It shows the HTTP code or the error of the mirror sync.
2. Examine the push mirror in Gitea. Make sure that it has correct access to GitHub.
3. Run `publish.sh --no-sync` to compare the branches and tags again.

## Tests

Run `tools/publish/test/run-tests.sh`. The tests make repositories in `/tmp` and use the example file with test names. The tests use local bare repositories as Gitea and GitHub. A test HTTP server works as the Gitea API. The tests do not use a host on a network.

The table shows each test and its result:

| Test | Result |
|---|---|
| An IP address of a local network (each range of RFC 1918 and the carrier-grade NAT range) | Stopped |
| A host name in the Tailscale network | Stopped |
| A name from the private rules file | Stopped |
| A UUID, a server path or a number after `monitor` | Stopped |
| A file of 6 MiB | Stopped |
| A `.gguf` file and each other file type that the file check stops | Stopped |
| A clean commit | Goes through |
| The example networks | Go through |
| The URL of the owner account and the license line | Go through |
| An IP address in upstream history, with the upstream base set | Not read |
| The same history, without the upstream base | Stopped |
| No private rules file | The hook stops the git push |
| `publish.sh` with an item | Exit code 1. The script does not use the API. |
| `publish.sh` mirror sync and step `d` | A run with no error and each run with an error |

If a private rules file is at the default path, the tests also use it. They make sure that the URL of the owner account, the license line and usual text go through. They make sure that the publication gate stops the name of the owner account in other shapes. They also make sure that the file has the mode 0600. Without the file, the tests do not do this part.

## Files

| File | Task |
|---|---|
| `pre-push` | The git hook |
| `install-hooks.sh` | Installs the hook and records checksums |
| `publish.sh` | The publication gate |
| `scan-dir.sh` | Scans a folder with the gitleaks rules of the publication gate |
| `lib.sh` | Functions that the other scripts use |
| `gitleaks.toml` | The gitleaks rules of the project. It adds to the default gitleaks rules. |
| `private-rules.example.toml` | Example of the private rules file |
| `allow-files.txt` | The allow-list |
| `test/run-tests.sh` | The tests |
| `test/fake-gitea.py` | A test server for the Gitea API |

## Limits

- The publication gate reads text. It does not read text that is part of an object in a file, for example in a `.png` file.
- The publication gate finds only the names that the gitleaks rules have. The private rules file does not have all names. A name that is not in the file goes through.
- The hook runs on your device. The flag `--no-verify` tells git not to run it. Because of this, `publish.sh` scans again.
- A new clone has no hook. Install the hook before the first git push.
- `scan-dir.sh` stops if the folder has a `.gitleaksignore` file.
