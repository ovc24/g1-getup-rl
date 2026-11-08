#!/usr/bin/env bash
# Before publishing: look for words that must not appear in the public repo (names, e-mails,
# user names, other projects...). The words are listed in notes/private_words.txt, one word or
# phrase per line (# = comment). notes/ is in .gitignore, so the list itself is never published.
#
#   bash scripts/check_public.sh            # exit code 1 if something was found
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIST="${1:-$REPO/notes/private_words.txt}"
cd "$REPO"
[ -f "$LIST" ] || { echo "No word list: create $LIST (one word or phrase per line)"; exit 1; }

words="$(mktemp)"
trap 'rm -f "$words"' EXIT
grep -v -E '^[[:space:]]*(#|$)' "$LIST" > "$words" || true
[ -s "$words" ] || { echo "The word list $LIST is empty"; exit 1; }

# The files a push would publish: tracked files + new files that are not ignored.
if git rev-parse --git-dir >/dev/null 2>&1; then
  mapfile -t files < <(git ls-files -co --exclude-standard)
else
  mapfile -t files < <(find . -type f -not -path './notes/*' -not -path './.git/*' | sed 's#^\./##')
fi

found=0
for f in "${files[@]}"; do
  [ -f "$f" ] || continue
  if grep -I -q . "$f" 2>/dev/null; then                      # text file: show the lines
    if out="$(grep -n -i -F -f "$words" "$f")"; then
      echo "$out" | sed "s#^#$f:#"
      found=1
    fi
  elif grep -a -q -i -F -f "$words" "$f"; then                # binary (onnx, pt): name only
    echo "$f: binary file contains a listed word"
    found=1
  fi
done

# E-mail addresses that commits publish (author and committer), and the one for new commits.
check_email() {
  case "$1" in
    *@users.noreply.github.com) ;;
    "") ;;
    *) echo "$2: $1 (use your GitHub no-reply address: GitHub > Settings > Emails)"; found=1 ;;
  esac
}
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  while read -r email; do check_email "$email" "commit e-mail"; done \
    < <(git log --format='%ae%n%ce' | sort -u)
fi
check_email "$(git config user.email 2>/dev/null || true)" "git config user.email"

# Names that commits publish (author, committer) and commit messages; the name for new commits.
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  if out="$(git log --format='%an%n%cn%n%B' | grep -n -i -F -f "$words")"; then
    echo "commit names or messages:"; echo "$out"
    found=1
  fi
fi
name="$(git config user.name 2>/dev/null || true)"
if [ -n "$name" ] && printf '%s\n' "$name" | grep -q -i -F -f "$words"; then
  echo "git config user.name: $name (use your GitHub user name: git config user.name <user>)"
  found=1
fi

if [ "$found" -eq 0 ]; then
  echo "OK: nothing from $(basename "$LIST") in the files to publish"
else
  echo "Fix the lines above before git push"
  exit 1
fi
