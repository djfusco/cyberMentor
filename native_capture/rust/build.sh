#!/bin/sh
# Builds cyberalfred-capture (Rust) and signs it with a stable code-signing
# identity, mirroring native_capture/scripts/build.sh.
#
# Ad-hoc signed binaries get a new signature hash on every build, which can
# invalidate macOS TCC permission grants (Screen Recording). Signing with a
# persistent local certificate + fixed identifier keeps those grants intact
# across rebuilds.
#
# One-time setup: create a self-signed "Code Signing" certificate named
# mentor-capture-dev in Keychain Access (Certificate Assistant > Create a
# Certificate > Identity Type: Self Signed Root > Certificate Type: Code
# Signing).
#
# Usage: build.sh [debug|release]

set -e

CONFIG="${1:-debug}"
SIGN_IDENTITY="mentor-capture-dev"
BUNDLE_ID="com.cyberalfred.mentor-capture"

cd "$(dirname "$0")/.."

echo "Building ($CONFIG)..."
if [ "$CONFIG" = "release" ]; then
  cargo build --release
else
  cargo build
fi

BINARY_PATH="target/$CONFIG/cyberalfred-capture"

echo "Signing $BINARY_PATH with identity '$SIGN_IDENTITY'..."
codesign --force --sign "$SIGN_IDENTITY" \
  --identifier "$BUNDLE_ID" \
  "$BINARY_PATH"

echo "Done. Signature:"
codesign -dvvv "$BINARY_PATH" 2>&1 | grep -E "Identifier|Authority|Signature"
