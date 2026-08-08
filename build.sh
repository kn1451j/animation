#!/bin/sh
# -O2 matters a lot here: stb_image's PNG decoder runs ~2.5x faster than at -O0,
# and startup decodes a couple thousand frames.
set -e
clang++ -std=c++17 -O2 main.cpp glad/glad.o -o anim \
    -Iglad/include -I/opt/homebrew/include -L/opt/homebrew/lib \
    -lglfw -lavcodec -lavformat -lavutil -lswscale -lswresample \
    -framework OpenGL

# Wrap the same binary in a double-clickable bundle. The 7 GB of assets are NOT
# copied in — the app finds them beside itself (see chdirToAssetRoot), so
# Anim.app has to stay in this folder, next to audios/, renders/ and vids/.
APP=Anim.app
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp anim "$APP/Contents/MacOS/anim"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>Anim</string>
  <key>CFBundleDisplayName</key>       <string>Anim</string>
  <key>CFBundleExecutable</key>        <string>anim</string>
  <key>CFBundleIdentifier</key>        <string>local.anim</string>
  <key>CFBundlePackageType</key>       <string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key>           <string>1</string>
  <!-- Retina: without this the window is scaled up from 1x and looks soft. -->
  <key>NSHighResolutionCapable</key>   <true/>
  <!-- Full-screen piece: no menu bar or Dock over it. -->
  <key>LSUIPresentationMode</key>      <integer>3</integer>
</dict>
</plist>
PLIST

# Locally built binaries carry no quarantine flag, but ad-hoc signing keeps
# Gatekeeper quiet if the bundle ever gets copied or zipped.
codesign --force --deep --sign - "$APP" 2>/dev/null || true
touch "$APP"   # nudge Finder to re-read the bundle
echo "built ./anim and ./$APP"
