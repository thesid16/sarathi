#!/bin/bash
#
# Install Sarathi on an Android phone, complete, in one command.
#
#   ./install-on-phone.sh
#
# Installs the app and copies the 2.4 GB scene-description model into the
# folder the app reads. Takes a couple of minutes, most of it the model.
#
# Why the model is not inside the APK: it is 2.4 GB. Bundling it would make
# the install larger than the free space on many of the phones this is for,
# and the app is completely usable without it - everything except the
# "Describe scene" button works exactly the same.
#
# Why not `run-as`: that only works on debuggable builds. The release APK is
# not debuggable, so the obvious `adb shell run-as ... cp` recipe fails with
# "package not debuggable" and the app then reports the model as missing while
# the file sits in /data/local/tmp. App-specific external storage needs no
# permission, takes a plain `adb push`, and is visible over USB - so it works
# for people who have adb and people who do not.

set -u
cd "$(dirname "$0")"

APK="sarathi.apk"
MODEL="models/weights/gemma-4-E2B-it.litertlm"
PKG="in.sarathi.app"
DEST="/sdcard/Android/data/$PKG/files/models"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
warn() { printf '  ! %s\n' "$*"; }

command -v adb >/dev/null || {
  warn "adb is not installed."
  warn "Install Android platform-tools, or copy $APK to the phone by hand"
  warn "and open it there - see INSTALL.md."
  exit 1
}

say "Looking for a phone"
adb start-server >/dev/null 2>&1
DEVICES=$(adb devices | grep -cw device)
if [ "$DEVICES" -eq 0 ]; then
  warn "No phone found. Connect it by USB, unlock it, and turn on"
  warn "Settings -> Developer options -> USB debugging."
  warn "Then accept the 'Allow USB debugging?' prompt on the phone."
  exit 1
fi
adb devices | grep -w device | grep -v List | sed 's/^/  /'

say "Installing the app"
if ! adb install -r "$APK"; then
  warn "Install failed. If it says the signature does not match, an older"
  warn "Sarathi is present: adb uninstall $PKG, then run this again."
  exit 1
fi

if [ -f "$MODEL" ]; then
  say "Copying the scene-description model (2.4 GB, about a minute)"
  adb shell mkdir -p "$DEST"
  # Launch once first so the external data directory exists and is owned
  # correctly; on a fresh install it does not exist until the app has run.
  adb shell monkey -p "$PKG" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1
  sleep 2
  adb shell mkdir -p "$DEST"
  if adb push "$MODEL" "$DEST/"; then
    printf '  copied to %s\n' "$DEST"
  else
    warn "Could not copy the model. The app will still work; the"
    warn "'Describe scene' button will explain where to put the file."
  fi
else
  warn "No model at $MODEL - skipping."
  warn "Everything except 'Describe scene' will work."
fi

say "Done"
cat <<'EOF'
  Open Sarathi on the phone and allow the camera. It starts guiding by itself.

    volume down          start / stop
    volume up, tapped    describe the scene
    volume up, held      read any text

  If it feels slow, unplug the phone - charging alone puts the CPU near its
  thermal limit and the app deliberately slows down rather than making it
  worse.
EOF
