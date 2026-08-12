#!/bin/bash
#
# Build the "full" APK: the app with the 2.4 GB scene-description model inside
# it, so there is exactly one file to hand over.
#
#   tools/bundle-model.sh          # full APK, ~2.5 GB
#   tools/bundle-model.sh --clean  # remove the staged model, back to 53 MB
#
# Why the model is split into three parts:
#
#   The Android build tool reads each asset into a JVM byte array, and the
#   maximum length of one is Integer.MAX_VALUE. A single 2.4 GB asset fails the
#   build with "Required array size too large" before anything is packaged.
#
# Why the parts keep the .litertlm extension:
#
#   `noCompress` matches on extension. Parts named `.partaa` are not matched,
#   so the packager tries to DEFLATE 1.2 GB in memory and dies with "Java heap
#   space". Named `foo.part01.litertlm` they are stored, and nothing large ever
#   goes through the heap.

set -eu
cd "$(dirname "$0")/.."
ASSETS=android/app/src/main/assets/models
MODEL=models/weights/gemma-4-E2B-it.litertlm

if [ "${1:-}" = "--clean" ]; then
  rm -f "$ASSETS"/gemma-*.litertlm
  echo "Staged model removed. The next release build is the slim ~53 MB APK."
  exit 0
fi

[ -f "$MODEL" ] || { echo "No model at $MODEL"; exit 1; }

rm -f "$ASSETS"/gemma-*.litertlm
echo "Splitting $(du -h "$MODEL" | cut -f1) into three assets..."
split -b 1200m "$MODEL" "$ASSETS/gemma-tmp."
i=1
for f in "$ASSETS"/gemma-tmp.*; do
  mv "$f" "$ASSETS/gemma-4-E2B-it.part0$i.litertlm"
  i=$((i + 1))
done
ls -lh "$ASSETS"/gemma-*.litertlm

echo "Building..."
(cd android && ./gradlew assembleRelease -q)
APK=android/app/build/outputs/apk/release/app-release.apk
ls -lh "$APK"
echo
echo "The app unpacks these on first use of Describe scene, which needs about"
echo "2.4 GB of free space on the phone in addition to the install itself."
