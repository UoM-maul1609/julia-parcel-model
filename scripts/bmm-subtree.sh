#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/bmm-subtree.sh add  <bmm-git-url> [ref]
  scripts/bmm-subtree.sh pull <bmm-git-url> [ref]
  scripts/bmm-subtree.sh push <bmm-git-url> [ref]

BMM is kept at ./bmm using git subtree. The default ref is main.
For the initial add, ./bmm must not already be tracked as an ordinary copied
folder. If converting an existing repository, commit/remove that copied folder
first, then run the add command.
USAGE
}

[[ $# -ge 2 ]] || { usage; exit 2; }
action=$1
remote=$2
ref=${3:-main}

case "$action" in
  add)  git subtree add  --prefix=bmm "$remote" "$ref" --squash ;;
  pull) git subtree pull --prefix=bmm "$remote" "$ref" --squash ;;
  push) git subtree push --prefix=bmm "$remote" "$ref" ;;
  *) usage; exit 2 ;;
esac
