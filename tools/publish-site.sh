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

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

echo "Staging the site..."
cp docs/assets/report.html "$STAGE/index.html"
mkdir -p "$STAGE/demo"
cp web/index.html web/sarathi.js "$STAGE/demo/"
cp -r web/model "$STAGE/demo/"

# .nojekyll stops GitHub Pages running Jekyll over it, which would ignore any
# file or folder beginning with an underscore.
touch "$STAGE/.nojekyll"

python3 - "$STAGE" <<'PY'
import pathlib, sys
stage = pathlib.Path(sys.argv[1])

# Cross-links, added here rather than committed into the sources: the report
# and the demo are also read on their own, where a link to "demo/" would break.
p = stage / "index.html"
s = p.read_text()
if "Try it live" not in s:
    s = s.replace('    <div class="meta">', '''    <p style="margin-top:22px">
      <a href="demo/" style="display:inline-block;background:var(--accent-bg);color:#14181B;
         padding:11px 20px;border-radius:8px;text-decoration:none;font-weight:600;font-size:15px">
        Try it live in this browser &rarr;</a>
      <a href="https://github.com/thesid16/sarathi/releases/latest" style="display:inline-block;
         margin-left:10px;padding:11px 20px;border-radius:8px;text-decoration:none;font-size:15px;
         border:1px solid var(--rule)">Download the Android app</a>
    </p>
    <div class="meta">''', 1)
    p.write_text(s)

d = stage / "demo" / "index.html"
s = d.read_text()
if "Back to the engineering report" not in s:
    s = s.replace('<h1>Sarathi <span>सारथी · live demo</span></h1>',
                  '<h1><a href="../" style="text-decoration:none;color:inherit">Sarathi</a>\n'
                  '    <span>सारथी · live demo</span></h1>', 1)
    s = s.replace('  Runs entirely in this browser',
                  '  <a href="../">&larr; Back to the engineering report</a>.\n'
                  '  Runs entirely in this browser', 1)
    d.write_text(s)
PY

WORKTREE=$(mktemp -d)
git worktree add -f "$WORKTREE" -B gh-pages >/dev/null 2>&1
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
git worktree remove --force "$WORKTREE" >/dev/null 2>&1
rm -rf "$WORKTREE"

echo
echo "  https://thesid16.github.io/sarathi/        the report"
echo "  https://thesid16.github.io/sarathi/demo/   the live demo"
echo
echo "GitHub Pages takes a minute or two to rebuild."
