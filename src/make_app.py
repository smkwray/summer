#!/usr/bin/env python3
"""Build the macOS application bundle. Standard library only.

    python3 src/make_app.py [--dest DIR]

WHAT THIS FIXES, AND WHAT IT DOES NOT:

    Finder / Spotlight              summ'er, with the icon      FIXED
    window title                     summ'er                     already right
    menu bar / Dock / app switcher   Python                      NOT fixed

The last row needs the process running Tk to have its executable inside this
bundle. Homebrew's interpreter routes GUI startup through its own framework
`Python.app`, and Tk takes the application name from there.

Fixing that row means shipping a private interpreter inside the bundle, which is
what py2app and PyInstaller exist to do. Neither is required by this lightweight
checkout launcher.

The bundle deliberately holds no application code. Its executable runs
`src/summ_ui.py` from the checkout that built it, so code updates take effect
without rebuilding. Rebuild the bundle after moving the checkout.

Default destination is ~/Applications, which needs no privileges and is indexed
by Spotlight.
"""
from __future__ import annotations
import argparse, pathlib, plistlib, shlex, shutil, stat, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
APP_NAME = "summ'er"          # what macOS displays
BUNDLE = "summer.app"         # what the file is called; an apostrophe in a path
                              # is legal but a nuisance in every shell
IDENT = "app.summer.desktop"

LAUNCHER_TEMPLATE = """#!/bin/sh
# No application code lives here. Run the UI from the checkout that built this
# bundle so the app and engine stay on the same version.
set -e
PROJECT_ROOT=__PROJECT_ROOT__
UI="$PROJECT_ROOT/src/summ_ui.py"
if [ ! -f "$UI" ]; then
  osascript -e 'display alert "summ'"'"'er" message "The project checkout has moved. Rebuild the application bundle from its new location."'
  exit 1
fi
# A python3 with Tk. The engine needs >=3.9; Homebrew ships Tk separately as
# python-tk, so a python3 without it must not be selected silently.
for py in /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
  if [ -x "$py" ] && "$py" -c 'import tkinter' >/dev/null 2>&1; then
    exec "$py" "$UI"
  fi
done
osascript -e 'display alert "summ'"'"'er" message "No python3 with tkinter was found. Try: brew install python-tk@3.14"'
exit 1
"""


def launcher(project_root: pathlib.Path = ROOT) -> str:
    """Render a launcher bound to one local checkout without publishing it."""
    root = pathlib.Path(project_root).expanduser().resolve()
    return LAUNCHER_TEMPLATE.replace("__PROJECT_ROOT__", shlex.quote(str(root)))


def build(dest: pathlib.Path) -> pathlib.Path:
    app = dest / BUNDLE
    if app.exists():
        shutil.rmtree(app)
    macos = app / "Contents" / "MacOS"
    res = app / "Contents" / "Resources"
    macos.mkdir(parents=True)
    res.mkdir(parents=True)

    exe = macos / "summer"
    exe.write_text(launcher())
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    icns = HERE / "assets" / "summer.icns"
    if icns.is_file():
        shutil.copy2(icns, res / "summer.icns")
    else:
        print("  no summer.icns — run make_icon.py first", file=sys.stderr)

    plist = {
        # Names the app in Finder, Spotlight and Raycast. It does NOT reach
        # the menu bar while the GUI runs under Homebrew's framework Python --
        # see the note at the top of this file.
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleExecutable": "summer",
        "CFBundleIdentifier": IDENT,
        "CFBundleIconFile": "summer",
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "11.0",
        "NSHighResolutionCapable": True,
        # Not a background agent: it owns a window and belongs in the Dock.
        "LSUIElement": False,
    }
    with (app / "Contents" / "Info.plist").open("wb") as fh:
        plistlib.dump(plist, fh)

    # Ad-hoc signature. Not distribution signing -- it has no Developer ID and
    # is not notarized -- but it gives the bundle a stable identity so macOS
    # stops re-prompting for permissions every time it is rebuilt.
    subprocess.run(["codesign", "--force", "--sign", "-", str(app)],
                   capture_output=True)
    # Nudge Launch Services so the new name and icon are picked up now rather
    # than whenever it next rescans.
    lsr = ("/System/Library/Frameworks/CoreServices.framework/Frameworks/"
           "LaunchServices.framework/Support/lsregister")
    if pathlib.Path(lsr).exists():
        subprocess.run([lsr, "-f", str(app)], capture_output=True)
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=str(pathlib.Path.home() / "Applications"))
    a = ap.parse_args()
    if sys.platform != "darwin":
        print("macOS only; on Windows the taskbar name is the window title, "
              "which is already correct.", file=sys.stderr)
        return 2
    dest = pathlib.Path(a.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    app = build(dest)
    print(f"  {app}")
    print(f"  Finder / Spotlight: {APP_NAME}")
    print(f"  menu bar / Dock: still 'Python' — needs a bundled interpreter")
    print(f"  open with:  open -a \"{app}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
