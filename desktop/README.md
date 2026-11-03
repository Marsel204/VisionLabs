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

From the repository root, run `./run.sh` (Linux) or `run.bat` (Windows) to start the current desktop and local API together. All launcher aliases use `app.main`; there is no alternate desktop implementation.

For browser-only development, start `python -m app.main --backend` in the project's virtual environment and run `npm run dev` here. The current pages live in `src/pages/`, shared studio components in `src/components/`, and bundled fonts in `public/fonts/`. Native application icons are generated from `public/favicon.svg` with `npm run tauri -- icon public/favicon.svg`.
