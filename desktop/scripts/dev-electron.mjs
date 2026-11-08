// Development launcher: Vite dev server + Electron pointed at it. The Python API is started by
// Electron itself (or reused if `python -m app.main --backend` is already running).
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import electron from 'electron';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const url = 'http://localhost:1420';

const vite = spawn(process.execPath, [path.join(root, 'node_modules/vite/bin/vite.js')], {
  cwd: root,
  stdio: 'inherit',
});

async function waitForVite() {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    if (vite.exitCode !== null) throw new Error('Vite exited before it was ready');
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      /* not up yet */
    }
    await new Promise(resolve => setTimeout(resolve, 150));
  }
  throw new Error(`Vite did not start at ${url}`);
}

function stopVite() {
  if (vite.exitCode === null) vite.kill();
}
process.on('SIGINT', () => { stopVite(); process.exit(130); });
process.on('SIGTERM', () => { stopVite(); process.exit(143); });

try {
  await waitForVite();
} catch (error) {
  console.error(error.message);
  stopVite();
  process.exit(1);
}

const app = spawn(electron, ['.'], {
  cwd: root,
  stdio: 'inherit',
  env: { ...process.env, ANOTARA_DEV_URL: url },
});
app.on('exit', code => {
  stopVite();
  process.exit(code ?? 0);
});
