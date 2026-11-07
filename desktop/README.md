# Anotara Desktop

## Removing a project

On **Proyek**, choose **Hapus proyek**, then confirm or select **Batal**. Removal only
clears the app's project history; images and annotations stay in the dataset folder.
Removing the active project also closes it and keeps it closed after restarting. Pending
annotation saves must finish first; failed saves and running jobs block removal. Import
the same folder again to reopen a removed project.

## Managing object classes

On **Impor dataset → Kelas objek**, use **Ubah** to rename a class and its saved annotations.
Use **Hapus** to remove an unused class, move its annotations to another class, or explicitly
delete its boxes and polygons. Changes require confirmation and automatically remap the other
YOLO class IDs; images are never deleted. Save pending annotation edits and finish running jobs
first. Originals are backed up under the dataset's `.visionlab/class-backups/<id>/` directory,
with a manifest recording the old/new registry and any newly created files. Keep these backups;
restoring one requires restoring all its files together, not just `data.yaml`.

## Verification

Run `npm test` for the component and workflow checks, and `npm run build` for the TypeScript
check and production bundle. `npm run test:canvas` runs the image/box alignment regression in
headless Chromium, including resize, zoom, pan, previews, and pointer editing. Install Chromium
on the test machine or set `CHROMIUM_BIN` to its executable path; no dataset or backend is needed.

## Development

The desktop app is an Electron shell (`electron/`) around this React UI. The main process opens
the window, starts the Python API (`electron/backend.mjs`), serves the built UI from the
`app://anotara` origin, and checks GitHub Releases for updates (`electron/updater.mjs`).

From the repository root, run `./run.sh` (Linux) or `run.bat` (Windows) to start the desktop and
local API together, or run `npm run electron:dev` here (Vite hot reload; the API is started for you
from the project's `.venv` unless one is already running on port 8765).

For browser-only development, start `python -m app.main --backend` and run `npm run dev` here.
The pages live in `src/pages/`, shared studio components in `src/components/`, and bundled fonts
in `public/fonts/`. App icons for the installers are in `build/` (`icon.png` 512px, `icon.ico`,
`icon.icns`).

## Packaging and releasing

```bash
# 1. Freeze the Python engine to ../dist/visionlab-backend (needs `uv sync --extra dev`)
uv run python scripts/build_backend.py
# 2. Build the UI and the installer for this OS into desktop/release/
npm ci && npm run package        # Windows: NSIS .exe, Linux: AppImage
```

`npm run package:dir` builds an unpacked app for quick checks. Installers are built per OS, so run
the steps on the OS you target (CI does this for Windows and Linux).

### Shipping an update

1. Push a version tag: `git tag v1.1.0 && git push origin v1.1.0`.
2. `.github/workflows/release.yml` tests, freezes the engine, builds the installers, sets the app
   version from the tag and publishes them with `latest.yml` / `latest-linux.yml` to the GitHub Release.
3. Installed apps check on startup and every 4 hours. The top bar shows **Pembaruan vX.Y.Z**; one
   click downloads it, then **Mulai ulang untuk memperbarui** restarts into the new version. A
   downloaded update also installs when the app is next quit. Help → *Periksa pembaruan…* checks on demand.

Updates only work in an installed app (Windows installer, Linux AppImage); development runs and
`.deb` installs show only the version. Set `ANOTARA_DISABLE_UPDATES=1` to turn updates off, or
`ANOTARA_UPDATE_FEED=<url>` to read `latest*.yml` from a staging or mirror server instead of GitHub.
Releases are not code-signed yet, so Windows SmartScreen warns on first install.
