import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { once } from 'node:events';
import { setTimeout as delay } from 'node:timers/promises';
import { test } from 'node:test';
import { createServer } from 'vite';

// Exercise browser layout and real pointer events; the hook harness cannot measure CSS.
test('review boxes and previews share the image geometry across resize, zoom and pan', { timeout: 60000 }, async t => {
  const server = await createServer({
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    define: { 'import.meta.env.VITE_API_URL': 'window.location.origin' },
    plugins: [{
      name: 'canvas-test-images',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const match = req.url?.match(/^\/api\/image\/(\d+)x(\d+)\.svg/);
          if (!match) return next();
          const [, w, h] = match;
          res.setHeader('Content-Type', 'image/svg+xml');
          res.end(`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}"><rect width="100%" height="100%" fill="#dae7f7"/><rect x="${w * .29}" y="${h * .27}" width="${w * .675}" height="${h * .55}" fill="#54745b"/></svg>`);
        });
      },
    }],
  });
  t.after(() => server.close());
  await server.listen();
  const origin = server.resolvedUrls.local[0];
  const profile = await mkdtemp(path.join(tmpdir(), 'anotara-canvas-'));
  const chrome = spawn(process.env.CHROMIUM_BIN || 'chromium', [
    '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
    '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe'] });
  t.after(async () => {
    if (chrome.exitCode === null) { chrome.kill(); await once(chrome, 'exit'); }
    await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  });
  const endpoint = await new Promise((resolve, reject) => {
    let output = '';
    chrome.once('error', reject);
    chrome.once('exit', () => reject(new Error(`Chromium exited before starting: ${output}`)));
    chrome.stderr.on('data', chunk => {
      output += chunk;
      const match = output.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) resolve(match[1]);
    });
  });
  const targets = await (await fetch(new URL('/json', endpoint.replace('ws:', 'http:')))).json();
  const ws = new WebSocket(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
  await once(ws, 'open');
  t.after(() => ws.close());
  let id = 0;
  const pending = new Map();
  ws.addEventListener('message', event => {
    const result = JSON.parse(event.data);
    if (!result.id) return;
    const call = pending.get(result.id);
    pending.delete(result.id);
    if (result.error) call.reject(new Error(result.error.message));
    else call.resolve(result.result);
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const call = ++id; pending.set(call, { resolve, reject });
    ws.send(JSON.stringify({ id: call, method, params }));
  });
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    assert.ok(!result.exceptionDetails, result.exceptionDetails?.exception?.description);
    return result.result.value;
  };
  const waitFor = async expression => {
    for (let attempt = 0; attempt < 150; attempt++) {
      if (await evaluate(expression)) return;
      await delay(50);
    }
    assert.fail(`Timed out: ${expression}`);
  };
  const viewport = async (width, height) => {
    await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
    await delay(150);
  };
  const geometry = () => evaluate(`(() => {
    const img = document.querySelector('#canvas-fixture img');
    const svg = img.nextElementSibling;
    const box = svg.querySelector('g rect');
    const rect = element => { const r = element.getBoundingClientRect(); return { x:r.x, y:r.y, width:r.width, height:r.height }; };
    return { image:rect(img), overlay:rect(svg), box:rect(box), polygon:rect(svg.querySelector('polygon')), viewport:rect(img.parentElement.parentElement) };
  })()`);
  const near = (actual, expected, label, tolerance = 1) => assert.ok(Math.abs(actual - expected) < tolerance, `${label}: ${actual} != ${expected}`);
  const aligned = (g, ratio, fitted = true) => {
    for (const key of ['x', 'y', 'width', 'height']) near(g.overlay[key], g.image[key], `overlay ${key}`);
    assert.ok(Math.abs(g.image.width / g.image.height - ratio) < .001, 'image keeps its aspect ratio');
    near(g.box.x, g.image.x + g.image.width * .29, 'box left');
    near(g.box.y, g.image.y + g.image.height * .27, 'box top');
    near(g.box.width, g.image.width * .675, 'box width');
    near(g.box.height, g.image.height * .55, 'box height');
    for (const key of ['x', 'y', 'width', 'height']) near(g.polygon[key], g.box[key], `polygon ${key}`);
    if (fitted) {
      assert.ok(g.image.x >= g.viewport.x && g.image.y >= g.viewport.y, 'full image starts inside viewport');
      assert.ok(g.image.x + g.image.width <= g.viewport.x + g.viewport.width + 1, 'full image fits width');
      assert.ok(g.image.y + g.image.height <= g.viewport.y + g.viewport.height + 1, 'full image fits height');
    }
  };
  await send('Page.enable');
  for (const [width, height] of [[640, 640], [1280, 720], [480, 960]]) {
    await viewport(1378, 895);
    await send('Page.navigate', { url: `${origin}tests/browser/canvas-layout.html?width=${width}&height=${height}` });
    await waitFor(`document.querySelector('#canvas-fixture img')?.complete && document.querySelector('#canvas-fixture img')?.naturalWidth > 0`);
    await delay(150);
    aligned(await geometry(), width / height);
    for (const [vw, vh] of [[1600, 1000], [780, 850], [1100, 650]]) {
      await viewport(vw, vh);
      aligned(await geometry(), width / height);
    }
    const preview = await evaluate(`(() => {
      const svg = document.querySelector('#preview-fixture svg');
      const rect = element => { const r = element.getBoundingClientRect(); return {x:r.x,y:r.y,width:r.width,height:r.height}; };
      return { image:rect(document.querySelector('#preview-fixture image, #preview-fixture img')), box:rect(svg.querySelector('g rect')) };
    })()`);
    assert.ok(Math.abs(preview.image.width / preview.image.height - width / height) < .001, 'preview keeps its aspect ratio');
    near(preview.box.x, preview.image.x + preview.image.width * .29, 'preview box left');
    near(preview.box.y, preview.image.y + preview.image.height * .27, 'preview box top');
    near(preview.box.width, preview.image.width * .675, 'preview box width');
    near(preview.box.height, preview.image.height * .55, 'preview box height');
  }

  await viewport(1378, 895);
  await send('Page.navigate', { url: `${origin}tests/browser/canvas-layout.html` });
  await waitFor(`document.querySelector('#canvas-fixture img')?.complete && document.querySelector('#canvas-fixture img')?.naturalWidth > 0`);
  await delay(150);
  const original = await geometry();
  const clickControl = async selector => {
    await evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
    await delay(150);
  };
  await clickControl('button[aria-label="Perbesar"]');
  const zoomed = await geometry();
  aligned(zoomed, 1, false);
  near(zoomed.image.width, original.image.width * 1.1, 'zoom scales both layers');
  await clickControl('button[title="Geser kanvas (H)"]');
  const drag = async (x, y, dx, dy) => {
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y });
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 });
    await delay(30);
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: x + dx, y: y + dy, button: 'left', buttons: 1 });
    await delay(30);
    await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: x + dx, y: y + dy, button: 'left', clickCount: 1 });
    await delay(150);
  };
  await drag(zoomed.image.x + 20, zoomed.image.y + 20, 20, 15);
  const panned = await geometry();
  aligned(panned, 1, false);
  near(panned.image.x, zoomed.image.x + 20, 'pan moves both layers horizontally');
  near(panned.image.y, zoomed.image.y + 15, 'pan moves both layers vertically');
  await clickControl('button[title="Pilih & ubah (V)"]');
  const pixels = panned.image.width / 640;
  await drag(panned.box.x + panned.box.width / 2, panned.box.y + panned.box.height / 2, 10 * pixels, 12 * pixels);
  const saved = await evaluate(`JSON.parse(document.querySelector('#boxes').textContent)[0]`);
  near(saved.x, 640 * .29 + 10, 'drag saves image-space x after zoom/pan');
  near(saved.y, 640 * .27 + 12, 'drag saves image-space y after zoom/pan');
  await clickControl('button[title="Kotak (B)"]');
  await drag(panned.image.x + 40 * pixels, panned.image.y + 40 * pixels, 60 * pixels, 70 * pixels);
  const drawn = await evaluate(`JSON.parse(document.querySelector('#boxes').textContent)[1]`);
  // Native pointer coordinates and saved pixel bounds are rounded at subpixel fit scales.
  near(drawn.x, 40, 'drawing starts at image-space x', 2);
  near(drawn.y, 40, 'drawing starts at image-space y', 2);
  near(drawn.width, 60, 'drawing saves image-space width', 2);
  near(drawn.height, 70, 'drawing saves image-space height', 2);
  await clickControl('button[title="Sesuaikan layar"]');
  const reset = await geometry();
  for (const key of ['x', 'y', 'width', 'height']) near(reset.image[key], original.image[key], `fit reset ${key}`);
});
