#!/bin/bash
#
# Double-click to run Sarathi on this Mac.
#
# Creates the virtualenv on first run, then starts the app. Anything passed on
# the command line is forwarded:
#
#   ./run-desktop.command --source walk.mp4 --speak --camera-pitch 20
#
# Exists because the repository path can contain a space, which makes the
# hand-typed command fail in a way that looks like the app is missing rather
# than like a quoting problem. A handover should not begin with a shell error.

cd "$(dirname "$0")/prototype" || {
  echo "Could not find the prototype folder next to this script."
  read -r -p "Press return to close."
  exit 1
}

if [ ! -x .venv/bin/python ]; then
  echo "First run - setting up Python packages. This takes a minute."
  echo
  if command -v uv >/dev/null 2>&1; then
    uv venv && uv pip install -e . || SETUP_FAILED=1
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m venv .venv && .venv/bin/pip install --quiet --upgrade pip \
      && .venv/bin/pip install -e . || SETUP_FAILED=1
  else
    echo "Python 3 is not installed. Install Python 3.12 or newer and try again."
    read -r -p "Press return to close."
    exit 1
  fi
  if [ -n "${SETUP_FAILED:-}" ] || [ ! -x .venv/bin/python ]; then
    echo
    echo "Setup failed. The error is above."
    read -r -p "Press return to close."
    exit 1
  fi
  echo
  echo "Setup complete."
  echo
fi

echo "Sarathi - starting on $(.venv/bin/python --version 2>&1)"
echo "Press Start in the window. Close the window to quit."
echo

.venv/bin/python -m sarathi.desktop "$@"
status=$?

# Hold the terminal open on a crash so the reason is readable, rather than the
# window vanishing and taking the traceback with it.
if [ $status -ne 0 ]; then
  echo
  echo "Sarathi exited with status $status - the error is above."
  read -r -p "Press return to close."
fi
