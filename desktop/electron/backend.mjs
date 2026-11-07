// Owns the local Python API process that the Anotara UI talks to.
import { EventEmitter } from 'node:events';
import { execFile, spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

export const BACKEND_HOST = '127.0.0.1';
export const BACKEND_PORT = 8765;
export const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;

const START_TIMEOUT_MS = 180_000; // A frozen PyTorch stack can take a while on the first launch.
const STOP_TIMEOUT_MS = 10_000;

/** True when an Anotara API answers on the fixed local port. */
export async function sessionReady(timeoutMs = 1000) {
  try {
    const response = await fetch(`${BACKEND_URL}/api/session`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok) return false;
    const body = await response.json();
    return typeof body?.token === 'string' && body.token.length > 0;
  } catch {
    return false;
  }
}

/**
 * Decide how to start the API.
 *  - ANOTARA_BACKEND_CMD (JSON array) overrides everything, for tests and debugging.
 *  - Packaged builds run the frozen backend shipped in the app's resources.
 *  - Development runs the project's Python environment from the repository root.
 */
export function resolveBackendCommand({ packaged, resourcesPath, repoRoot, platform = process.platform, env = process.env }) {
  if (env.ANOTARA_BACKEND_CMD) {
    const parsed = JSON.parse(env.ANOTARA_BACKEND_CMD);
    if (!Array.isArray(parsed) || parsed.length === 0 || parsed.some(item => typeof item !== 'string')) {
      throw new Error('ANOTARA_BACKEND_CMD must be a non-empty JSON array of strings');
    }
    return { command: parsed[0], args: parsed.slice(1), cwd: repoRoot };
  }
  if (packaged) {
    const directory = path.join(resourcesPath, 'backend');
    const executable = path.join(directory, platform === 'win32' ? 'visionlab-backend.exe' : 'visionlab-backend');
    if (!fs.existsSync(executable)) {
      throw new Error(`The bundled Anotara engine is missing: ${executable}`);
    }
    return { command: executable, args: ['--backend'], cwd: directory };
  }
  const python = path.join(repoRoot, '.venv', platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  if (fs.existsSync(python)) {
    return { command: python, args: ['-m', 'app.main', '--backend'], cwd: repoRoot };
  }
  return { command: 'uv', args: ['run', 'python', '-m', 'app.main', '--backend'], cwd: repoRoot };
}

export class BackendManager extends EventEmitter {
  /** @param {{ logFile: string, resolve: () => ReturnType<typeof resolveBackendCommand> }} options */
  constructor({ logFile, resolve }) {
    super();
    this.logFile = logFile;
    this.resolve = resolve;
    this.child = null;
    this.exited = true;
    this.startedByUs = false;
    this.stopping = false;
  }

  /** Start the API (or adopt one that is already running) and wait until it answers. */
  async start() {
    if (await sessionReady()) {
      this.startedByUs = false;
      return;
    }
    const { command, args, cwd } = this.resolve();
    fs.mkdirSync(path.dirname(this.logFile), { recursive: true });
    const log = fs.openSync(this.logFile, 'a');
    fs.writeSync(log, `\n--- ${new Date().toISOString()} starting: ${command} ${args.join(' ')}\n`);
    try {
      this.stopping = false;
      this.exited = false;
      this.startedByUs = true;
      this.child = spawn(command, args, {
        cwd,
        env: { ...process.env, PYTHONUNBUFFERED: '1', PYTORCH_CUDA_ALLOC_CONF: process.env.PYTORCH_CUDA_ALLOC_CONF || 'expandable_segments:True' },
        stdio: ['ignore', log, log],
        windowsHide: true,
      });
    } finally {
      fs.closeSync(log); // The child owns its own copy of the descriptor.
    }
    const child = this.child;
    const early = new Promise((_, reject) => {
      child.once('error', error => {
        this.exited = true;
        reject(new Error(`Could not start the Anotara engine (${command}): ${error.message}`));
      });
      child.once('exit', (code, signal) => {
        this.exited = true;
        if (this.child === child) this.child = null;
        const reason = signal ? `signal ${signal}` : `code ${code}`;
        if (this.stopping) return;
        this.emit('crashed', reason);
        reject(new Error(`The Anotara engine exited unexpectedly (${reason}).`));
      });
    });
    // Rejections after readiness are surfaced through the 'crashed' event instead.
    early.catch(() => {});
    await Promise.race([this.#waitUntilReady(), early]);
  }

  async #waitUntilReady() {
    const deadline = Date.now() + START_TIMEOUT_MS;
    while (Date.now() < deadline) {
      if (this.exited) return new Promise(() => {}); // `early` reports the failure.
      if (await sessionReady()) return;
      await new Promise(resolve => setTimeout(resolve, 300));
    }
    throw new Error('The Anotara engine did not become ready in time.');
  }

  /** Stop only a process this manager started. Safe to call repeatedly. */
  async stop() {
    const child = this.child;
    if (!child || this.exited || !this.startedByUs) return;
    this.stopping = true;
    const gone = new Promise(resolve => child.once('exit', resolve));
    if (process.platform === 'win32') {
      // Kill the whole tree: the frozen engine may have spawned workers.
      execFile('taskkill', ['/pid', String(child.pid), '/T', '/F'], () => {});
    } else {
      child.kill('SIGTERM');
    }
    const timer = setTimeout(() => {
      if (!this.exited) child.kill('SIGKILL');
    }, STOP_TIMEOUT_MS);
    await gone;
    clearTimeout(timer);
  }
}
