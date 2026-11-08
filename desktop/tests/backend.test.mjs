import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { resolveBackendCommand } from '../electron/backend.mjs';

const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'anotara-backend-'));

test('packaged builds run the frozen engine from the app resources', () => {
  const resources = fs.mkdtempSync(path.join(os.tmpdir(), 'anotara-res-'));
  fs.mkdirSync(path.join(resources, 'backend'));
  fs.writeFileSync(path.join(resources, 'backend', 'visionlab-backend'), '');
  const result = resolveBackendCommand({ packaged: true, resourcesPath: resources, repoRoot, platform: 'linux', env: {} });
  assert.equal(result.command, path.join(resources, 'backend', 'visionlab-backend'));
  assert.deepEqual(result.args, ['--backend']);
  assert.equal(result.cwd, path.join(resources, 'backend'));
});

test('a packaged build without its engine fails with a clear message', () => {
  assert.throws(
    () => resolveBackendCommand({ packaged: true, resourcesPath: repoRoot, repoRoot, platform: 'win32', env: {} }),
    /visionlab-backend\.exe/,
  );
});

test('development prefers the project virtualenv and falls back to uv', () => {
  const fallback = resolveBackendCommand({ packaged: false, resourcesPath: '', repoRoot, platform: 'linux', env: {} });
  assert.equal(fallback.command, 'uv');
  assert.deepEqual(fallback.args, ['run', 'python', '-m', 'app.main', '--backend']);

  fs.mkdirSync(path.join(repoRoot, '.venv', 'bin'), { recursive: true });
  fs.writeFileSync(path.join(repoRoot, '.venv', 'bin', 'python'), '');
  const venv = resolveBackendCommand({ packaged: false, resourcesPath: '', repoRoot, platform: 'linux', env: {} });
  assert.equal(venv.command, path.join(repoRoot, '.venv', 'bin', 'python'));
  assert.deepEqual(venv.args, ['-m', 'app.main', '--backend']);
});

test('ANOTARA_BACKEND_CMD overrides the command and is validated', () => {
  const override = resolveBackendCommand({
    packaged: true, resourcesPath: '', repoRoot, env: { ANOTARA_BACKEND_CMD: JSON.stringify(['node', 'fake.js', '--x']) },
  });
  assert.equal(override.command, 'node');
  assert.deepEqual(override.args, ['fake.js', '--x']);
  assert.throws(() => resolveBackendCommand({ packaged: false, repoRoot, env: { ANOTARA_BACKEND_CMD: '"node"' } }), /JSON array/);
});
