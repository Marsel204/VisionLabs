import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { test } from 'node:test';
import { UpdateController, plainReleaseNotes } from '../electron/updater.mjs';

const quiet = { info() {}, warn() {}, error() {} };

/** Minimal stand-in for electron-updater's autoUpdater. */
function fakeUpdater() {
  const updater = new EventEmitter();
  updater.calls = [];
  updater.checkForUpdates = async () => {
    updater.calls.push('check');
    updater.emit('checking-for-update');
    if (updater.nextCheck) await updater.nextCheck(updater);
  };
  updater.downloadUpdate = async () => {
    updater.calls.push('download');
    updater.emit('download-progress', { percent: 42.4 });
    updater.emit('update-downloaded', { version: '1.2.0' });
  };
  updater.quitAndInstall = (silent, forceRun) => updater.calls.push(`install:${silent}:${forceRun}`);
  return updater;
}

function controller(overrides = {}) {
  const updater = overrides.updater ?? fakeUpdater();
  return { updater, updates: new UpdateController({ updater, currentVersion: '1.0.0', logger: quiet, ...overrides }) };
}

test('updates are opt-in downloads that install on the next quit', () => {
  const { updater } = controller();
  assert.equal(updater.autoDownload, false);
  assert.equal(updater.autoInstallOnAppQuit, true);
  assert.equal(updater.allowDowngrade, false);
});

test('a disabled controller never touches the updater', async () => {
  const { updater, updates } = controller({ disabledReason: 'development' });
  assert.equal(updates.getState().status, 'disabled');
  await updates.check({ manual: true });
  await updates.download();
  await updates.install();
  assert.deepEqual(updater.calls, []);
});

test('check -> available -> download -> downloaded -> install, in order', async () => {
  let installed = false;
  const { updater, updates } = controller({ beforeInstall: async () => { installed = true; } });
  updater.nextCheck = u => u.emit('update-available', { version: '1.2.0', releaseNotes: '<p>Faster export</p>' });
  const seen = [];
  updates.onChange(state => seen.push(state.status));

  await updates.check({ manual: true });
  assert.equal(updates.getState().status, 'available');
  assert.equal(updates.getState().availableVersion, '1.2.0');
  assert.equal(updates.getState().releaseNotes, 'Faster export');

  await updates.download();
  assert.equal(updates.getState().status, 'downloaded');
  assert.ok(seen.includes('downloading'));

  await updates.install();
  assert.ok(installed, 'the engine must be stopped before the installer runs');
  assert.deepEqual(updater.calls, ['check', 'download', 'install:true:true']);
});

test('install and download are ignored in the wrong state', async () => {
  const { updater, updates } = controller();
  await updates.download();
  await updates.install();
  assert.deepEqual(updater.calls, []);
});

test('a manual check that finds nothing reports up-to-date', async () => {
  const { updater, updates } = controller();
  updater.nextCheck = u => u.emit('update-not-available', { version: '1.0.0' });
  await updates.check({ manual: true });
  assert.equal(updates.getState().status, 'up-to-date');
  assert.ok(updates.getState().checkedAt);
});

test('a failed background check stays quiet but a manual one reports', async () => {
  const { updater, updates } = controller();
  updater.nextCheck = () => { throw new Error('getaddrinfo ENOTFOUND github.com\nstack...'); };
  await updates.check();
  assert.equal(updates.getState().status, 'idle');
  assert.equal(updates.getState().error, null);

  await updates.check({ manual: true });
  assert.equal(updates.getState().status, 'error');
  assert.equal(updates.getState().error, 'getaddrinfo ENOTFOUND github.com');
});

test('a failed download can be retried', async () => {
  const { updater, updates } = controller();
  updater.nextCheck = u => u.emit('update-available', { version: '1.2.0' });
  await updates.check({ manual: true });
  const original = updater.downloadUpdate;
  updater.downloadUpdate = async () => { throw new Error('connection reset'); };
  await updates.download();
  assert.equal(updates.getState().status, 'error');
  assert.equal(updates.getState().availableVersion, '1.2.0');

  updater.downloadUpdate = original;
  await updates.download();
  assert.equal(updates.getState().status, 'downloaded');
});

test('overlapping checks are ignored while one is running', async () => {
  const { updater, updates } = controller();
  let release;
  updater.nextCheck = () => new Promise(resolve => { release = resolve; });
  const first = updates.check({ manual: true });
  await updates.check({ manual: true });
  assert.equal(updater.calls.length, 1);
  release();
  await first;
});

test('release notes become plain text', () => {
  assert.equal(plainReleaseNotes(null), null);
  assert.equal(plainReleaseNotes('<h2>New</h2><ul><li>One</li><li>Two &amp; three</li></ul>'), 'New\nOne\nTwo & three');
  assert.equal(plainReleaseNotes([{ note: 'First' }, { note: 'Second' }]), 'First\n\nSecond');
});
