# React + TypeScript + Vite

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

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend enabling type-aware lint rules by installing `oxlint-tsgolint` and editing `.oxlintrc.json`:

```json
{
  "$schema": "./node_modules/oxlint/configuration_schema.json",
  "plugins": ["react", "typescript", "oxc"],
  "options": {
    "typeAware": true
  },
  "rules": {
    "react/rules-of-hooks": "error",
    "react/only-export-components": ["warn", { "allowConstantExport": true }]
  }
}
```

See the [Oxlint rules documentation](https://oxc.rs/docs/guide/usage/linter/rules) for the full list of rules and categories.
