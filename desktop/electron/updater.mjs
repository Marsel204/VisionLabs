// Update state machine around electron-updater. It is kept free of Electron imports so it can be
// unit-tested with a fake updater; main.mjs wires it to the real one and to the renderer.

export const CHECK_DELAY_MS = 15_000;
export const CHECK_INTERVAL_MS = 4 * 60 * 60 * 1000;

/** @typedef {'disabled'|'idle'|'checking'|'up-to-date'|'available'|'downloading'|'downloaded'|'error'} UpdateStatus */

/** Release notes arrive as HTML, markdown, or a list of entries; the UI only shows plain text. */
export function plainReleaseNotes(notes) {
  if (!notes) return null;
  const text = Array.isArray(notes) ? notes.map(entry => entry?.note ?? '').join('\n\n') : String(notes);
  const cleaned = text
    .replace(/<\s*br\s*\/?>/gi, '\n')
    .replace(/<\/(p|li|h\d)>/gi, '\n')
    .replace(/<[^>]*>/g, '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  return cleaned || null;
}

export class UpdateController {
  /**
   * @param {object} options
   * @param {import('electron-updater').AppUpdater} options.updater
   * @param {string} options.currentVersion
   * @param {string|null} [options.disabledReason] Set to turn updates off (development, deb installs, ...).
   * @param {() => Promise<void>} [options.beforeInstall] Runs before the app quits to install.
   * @param {{info: Function, warn: Function, error: Function}} [options.logger]
   */
  constructor({ updater, currentVersion, disabledReason = null, beforeInstall = async () => {}, logger = console }) {
    this.updater = updater;
    this.beforeInstall = beforeInstall;
    this.logger = logger;
    this.listeners = new Set();
    this.timers = [];
    this.manualCheck = false;
    this.state = {
      status: disabledReason ? 'disabled' : 'idle',
      currentVersion,
      availableVersion: null,
      releaseNotes: null,
      percent: 0,
      checkedAt: null,
      error: null,
      disabledReason,
    };
    if (disabledReason) return;

    // Downloading is an explicit user action (the engine makes releases large); once downloaded the
    // update installs on the next quit even if the user never presses "restart".
    updater.autoDownload = false;
    updater.autoInstallOnAppQuit = true;
    updater.allowDowngrade = false;
    updater.logger = logger;

    updater.on('checking-for-update', () => this.#set({ status: 'checking', error: null }));
    updater.on('update-available', info => this.#set({
      status: 'available',
      availableVersion: info.version,
      releaseNotes: plainReleaseNotes(info.releaseNotes),
      percent: 0,
      checkedAt: Date.now(),
    }));
    updater.on('update-not-available', () => this.#set({
      status: 'up-to-date', availableVersion: null, releaseNotes: null, checkedAt: Date.now(),
    }));
    updater.on('download-progress', progress => this.#set({
      status: 'downloading', percent: Math.max(0, Math.min(100, Math.round(progress.percent ?? 0))),
    }));
    updater.on('update-downloaded', info => this.#set({
      status: 'downloaded', availableVersion: info.version, percent: 100,
    }));
    updater.on('error', error => this.#fail(error));
  }

  getState() {
    return { ...this.state };
  }

  /** Subscribe to state changes. Returns an unsubscribe function. */
  onChange(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Check once shortly after launch, then periodically. */
  startPolling() {
    if (this.state.status === 'disabled') return;
    const first = setTimeout(() => this.check(), CHECK_DELAY_MS);
    const repeat = setInterval(() => this.check(), CHECK_INTERVAL_MS);
    first.unref?.();
    repeat.unref?.();
    this.timers.push(first, repeat);
  }

  dispose() {
    for (const timer of this.timers) {
      clearTimeout(timer);
      clearInterval(timer);
    }
    this.timers = [];
    this.listeners.clear();
  }

  /** @param {{ manual?: boolean }} [options] Manual checks report failures; background ones stay quiet. */
  async check({ manual = false } = {}) {
    const { status } = this.state;
    if (status === 'disabled' || status === 'checking' || status === 'downloading' || status === 'downloaded') {
      return this.getState();
    }
    this.manualCheck = manual;
    try {
      await this.updater.checkForUpdates();
    } catch (error) {
      this.#fail(error);
    }
    return this.getState();
  }

  async download() {
    const { status, availableVersion } = this.state;
    const retry = status === 'error' && availableVersion;
    if (status !== 'available' && !retry) return this.getState();
    this.manualCheck = true;
    this.#set({ status: 'downloading', percent: 0, error: null });
    try {
      await this.updater.downloadUpdate();
    } catch (error) {
      this.#fail(error);
    }
    return this.getState();
  }

  async install() {
    if (this.state.status !== 'downloaded') return this.getState();
    try {
      await this.beforeInstall();
    } catch (error) {
      this.logger.warn?.(`Update: engine did not stop cleanly before install: ${error?.message ?? error}`);
    }
    // Silent, and relaunch afterwards: the user only sees the app restart on the new version.
    this.updater.quitAndInstall(true, true);
    return this.getState();
  }

  #fail(error) {
    const message = error?.message ? String(error.message).split('\n')[0] : String(error);
    this.logger.warn?.(`Update: ${message}`);
    if (this.manualCheck || this.state.status === 'downloading') {
      this.#set({ status: 'error', error: message });
    } else {
      // A failed background check (offline, rate limited) is not worth interrupting anyone for.
      this.#set({ status: this.state.availableVersion ? 'available' : 'idle', error: null });
    }
  }

  #set(patch) {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) listener(this.getState());
  }
}
