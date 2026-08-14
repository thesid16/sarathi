#!/bin/bash
#
# Publish the project site to https://thesid16.github.io/sarathi/
#
#   tools/publish-site.sh
#
# The site is the engineering report plus the browser demo, served from the
# `gh-pages` branch. That branch holds ONLY the built site - it is not a copy
# of the project - so this script rebuilds it from scratch each time rather
# than merging.
#
# Sources:
#   docs/assets/report.html  ->  index.html
#   web/                     ->  demo/

set -eu
cd "$(dirname "$0")/.."

# Clear any worktree left registered by an interrupted run. Without this the
# next run fails with "cannot force update the branch used by worktree at ...",
# which points at a directory that no longer exists.
git worktree prune

STAGE=$(mktemp -d)
WORKTREE=""
# Cleanup on ANY exit, not just success. `set -e` means a failed git command
# jumps straight out, and the tidy-up that used to sit at the end never ran -
# which is exactly how the stale worktree above got there.
cleanup() {
  [ -n "$WORKTREE" ] && git worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
  [ -n "$WORKTREE" ] && rm -rf "$WORKTREE"
  rm -rf "$STAGE"
  git worktree prune
}
trap cleanup EXIT

echo "Staging the site..."
cp docs/assets/report.html "$STAGE/index.html"
cp docs/assets/app-detecting.jpg docs/assets/app-doorway.jpg \
   docs/assets/app-stairs.jpg docs/assets/app-hallway.jpg "$STAGE/"
mkdir -p "$STAGE/technical"
cp docs/assets/technical/index.html "$STAGE/technical/"
mkdir -p "$STAGE/demo"
cp web/index.html web/sarathi.js "$STAGE/demo/"
cp -r web/model "$STAGE/demo/"

# .nojekyll stops GitHub Pages running Jekyll over it, which would ignore any
# file or folder beginning with an underscore.
touch "$STAGE/.nojekyll"

WORKTREE=$(mktemp -d)

# Base the worktree on the PUBLISHED branch, not on the current one.
#
# `git worktree add -B gh-pages` recreates the branch from wherever HEAD is -
# i.e. from main - so the resulting commit is not a descendant of what is
# already published and the push is rejected as non-fast-forward. It works
# exactly once, the first time, when there is nothing to be a descendant of.
git fetch -q origin gh-pages 2>/dev/null || true
if git rev-parse --verify -q origin/gh-pages >/dev/null; then
  git worktree add -f "$WORKTREE" -B gh-pages origin/gh-pages >/dev/null 2>&1
else
  git worktree add -f "$WORKTREE" --orphan gh-pages >/dev/null 2>&1 \
    || git worktree add -f "$WORKTREE" -B gh-pages >/dev/null 2>&1
fi
(
  cd "$WORKTREE"
  find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
  cp -r "$STAGE"/. .
  cat > README.md <<'MD'
# Sarathi — project site

Published at **https://thesid16.github.io/sarathi/**

- `index.html` — the engineering report
- `demo/` — the browser demo: same model and decoding as the phone, client-side

This branch holds only the published site. The project lives on `main`.
Regenerate it with `tools/publish-site.sh`.
MD
  git add -A
  if git diff --cached --quiet; then
    echo "No changes to publish."
  else
    git commit -q -m "Publish the project site"
    git push -q origin gh-pages
    echo "Published."
  fi
)
echo
echo "  https://thesid16.github.io/sarathi/        the report"
echo "  https://thesid16.github.io/sarathi/demo/   the live demo"
echo
echo "GitHub Pages takes a minute or two to rebuild."
