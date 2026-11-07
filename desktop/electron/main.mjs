// Anotara desktop shell: opens the window, runs the local Python API, and keeps the app updated.
import { app, BrowserWindow, ipcMain, Menu, net, protocol, shell } from 'electron';
import electronUpdater from 'electron-updater';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { BACKEND_URL, BackendManager, resolveBackendCommand } from './backend.mjs';
import { UpdateController } from './updater.mjs';

const { autoUpdater } = electronUpdater;

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');
const DIST_DIR = path.resolve(HERE, '..', 'dist');
const SPLASH = path.join(HERE, 'splash.html');
const APP_ORIGIN = 'app://anotara';
const DEV_URL = process.env.ANOTARA_DEV_URL || null;
const WINDOW_TITLE = 'Anotara - AI Annotation Studio';

app.setName('Anotara');
if (process.platform === 'win32') app.setAppUserModelId('com.visionlabs.anotara');

// The built UI is served from its own secure origin instead of file://, so the API can allow
// exactly one trusted origin (see ALLOWED_ORIGINS in app/api/server.py).
protocol.registerSchemesAsPrivileged([
  { scheme: 'app', privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true } },
]);

/** @type {BrowserWindow | null} */
let mainWindow = null;
let quitting = false;
let backend;
let updates;

function logsDirectory() {
  return app.getPath('logs');
}

function createBackend() {
  return new BackendManager({
    logFile: path.join(logsDirectory(), 'backend.log'),
    resolve: () => resolveBackendCommand({
      packaged: app.isPackaged,
      resourcesPath: process.resourcesPath,
      repoRoot: REPO_ROOT,
    }),
  });
}

function updateDisabledReason() {
  if (process.env.ANOTARA_DISABLE_UPDATES === '1') return 'Pembaruan dimatikan oleh ANOTARA_DISABLE_UPDATES.';
  if (!app.isPackaged) return 'Pembaruan hanya aktif pada aplikasi yang sudah terpasang.';
  if (process.platform === 'linux' && !process.env.APPIMAGE) {
    return 'Pembaruan otomatis di Linux hanya tersedia untuk paket AppImage.';
  }
  return null;
}

function broadcastUpdateState(state) {
  for (const window of BrowserWindow.getAllWindows()) {
    if (!window.isDestroyed()) window.webContents.send('anotara:update:state', state);
  }
}

async function serveAppFile(request) {
  const url = new URL(request.url);
  if (url.host !== 'anotara') return new Response('Not found', { status: 404 });
  let relative = decodeURIComponent(url.pathname).replace(/^\/+/, '') || 'index.html';
  let file = path.resolve(DIST_DIR, relative);
  // Never serve anything outside the built UI.
  if (file !== DIST_DIR && !file.startsWith(DIST_DIR + path.sep)) return new Response('Forbidden', { status: 403 });
  if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    // Unknown paths without an extension fall back to the single-page app shell.
    if (path.extname(relative)) return new Response('Not found', { status: 404 });
    file = path.join(DIST_DIR, 'index.html');
  }
  return net.fetch(pathToFileURL(file).toString());
}

function createWindow() {
  const window = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 640,
    minHeight: 480,
    show: false,
    title: WINDOW_TITLE,
    backgroundColor: '#f3f7fd',
    autoHideMenuBar: true,
    icon: path.join(HERE, '..', 'build', 'icon.png'),
    webPreferences: {
      preload: path.join(HERE, 'preload.cjs'),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
    },
  });
  window.once('ready-to-show', () => window.show());
  window.on('closed', () => { if (mainWindow === window) mainWindow = null; });
  // The page title is "Anotara"; keep the full product title in the title bar.
  window.on('page-title-updated', event => event.preventDefault());

  const trusted = origin => origin === APP_ORIGIN || (DEV_URL && origin === new URL(DEV_URL).origin);
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//i.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
  window.webContents.on('will-navigate', (event, url) => {
    const target = new URL(url);
    // Only the bundled splash may be a file:// page; a file dropped on the window must not navigate.
    if (target.protocol === 'file:' && path.resolve(fileURLToPath(target)) === SPLASH) return;
    if (trusted(target.origin)) return;
    event.preventDefault();
    if (/^https?:$/.test(target.protocol)) shell.openExternal(url);
  });
  return window;
}

function showSplash(window, error) {
  const query = error ? { error } : undefined;
  return window.loadFile(SPLASH, { query });
}

async function boot() {
  if (!mainWindow || mainWindow.isDestroyed()) mainWindow = createWindow();
  const window = mainWindow;
  await showSplash(window);
  try {
    await backend.start();
  } catch (error) {
    console.error(error);
    if (!window.isDestroyed()) await showSplash(window, `${error.message} Lihat backend.log di folder log.`);
    return;
  }
  if (window.isDestroyed()) return;
  await window.loadURL(DEV_URL ?? `${APP_ORIGIN}/index.html`);
}

function buildMenu() {
  const mac = process.platform === 'darwin';
  const template = [
    ...(mac ? [{ role: 'appMenu' }] : []),
    { role: 'fileMenu' },
    { role: 'editMenu' },
    { role: 'viewMenu' },
    { role: 'windowMenu' },
    {
      role: 'help',
      submenu: [
        { label: 'Periksa pembaruan…', click: () => updates.check({ manual: true }) },
        { label: 'Buka folder log', click: () => shell.openPath(logsDirectory()) },
        { label: 'Catatan rilis', click: () => shell.openExternal('https://github.com/Marsel204/VisionLabs/releases') },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function registerIpc() {
  ipcMain.handle('anotara:app-info', () => ({
    version: app.getVersion(),
    platform: process.platform,
    arch: process.arch,
    packaged: app.isPackaged,
    backendUrl: BACKEND_URL,
    logsPath: logsDirectory(),
  }));
  ipcMain.handle('anotara:open-logs', () => shell.openPath(logsDirectory()));
  ipcMain.handle('anotara:retry-start', () => boot());
  ipcMain.handle('anotara:update:get-state', () => updates.getState());
  ipcMain.handle('anotara:update:check', () => updates.check({ manual: true }));
  ipcMain.handle('anotara:update:download', () => updates.download());
  ipcMain.handle('anotara:update:install', () => updates.install());
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (!mainWindow) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  });

  app.whenReady().then(async () => {
    protocol.handle('app', serveAppFile);
    backend = createBackend();
    // Optional override of the update source (a staging channel or an internal mirror serving
    // latest.yml / latest-linux.yml). By default the GitHub Releases feed from electron-builder.yml is used.
    if (process.env.ANOTARA_UPDATE_FEED) {
      autoUpdater.setFeedURL({ provider: 'generic', url: process.env.ANOTARA_UPDATE_FEED });
    }
    updates = new UpdateController({
      updater: autoUpdater,
      currentVersion: app.getVersion(),
      disabledReason: updateDisabledReason(),
      beforeInstall: async () => {
        quitting = true;
        await backend.stop();
      },
    });
    updates.onChange(broadcastUpdateState);
    backend.on('crashed', reason => {
      console.error(`Anotara engine stopped (${reason})`);
      if (quitting || !mainWindow || mainWindow.isDestroyed()) return;
      showSplash(mainWindow, `Mesin AI berhenti tiba-tiba (${reason}). Lihat backend.log di folder log.`);
    });
    registerIpc();
    buildMenu();
    await boot();
    updates.startPolling();
  });

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0 && backend) boot();
  });

  // Stop the engine we started before exiting, then let the quit proceed.
  app.on('before-quit', event => {
    if (quitting || !backend) return;
    quitting = true;
    event.preventDefault();
    backend.stop().finally(() => app.quit());
  });
}
