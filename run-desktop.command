#!/bin/bash
#
# Double-click to run Sarathi on this Mac.
#
# Exists because the repository path contains a space, which makes the
# hand-typed command fail in a way that looks like the app is missing rather
# than like a quoting problem. A demo should not begin with a shell error.
#
# Anything passed on the command line is forwarded, so this still works:
#   ./run-desktop.command --source walk.mp4 --speak

cd "$(dirname "$0")/prototype" || exit 1

if [ ! -x .venv/bin/python ]; then
  echo "No virtualenv found at $(pwd)/.venv"
  echo
  echo "Create it with:"
  echo "  cd \"$(pwd)\" && uv venv && uv pip install -e ."
  echo
  read -r -p "Press return to close."
  exit 1
fi

echo "Sarathi — starting on $(.venv/bin/python --version 2>&1)"
echo "Press Start in the window. Close the window to quit."
echo

.venv/bin/python -m sarathi.desktop "$@"
status=$?

# Hold the terminal open on a crash so the reason is readable, rather than the
# window vanishing and taking the traceback with it.
if [ $status -ne 0 ]; then
  echo
  echo "Sarathi exited with status $status — the error is above."
  read -r -p "Press return to close."
fi
