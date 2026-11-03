// The icon font is a subset; an icon missing from it renders as its raw name (e.g. "space_dashboard").
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { test } = require('node:test');

const src = path.resolve(__dirname, '../src');
const subset = new Set(JSON.parse(fs.readFileSync(path.resolve(__dirname, '../public/fonts/icon-names.json'), 'utf8')));

function files(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return files(full);
    return /\.(ts|tsx)$/.test(entry.name) ? [full] : [];
  });
}

/** Icon names in the places the code uses them (props, object fields, toasts, raw icon spans). */
function usedIcons(text) {
  const found = new Set();
  const add = value => { if (/^[a-z][a-z0-9_]*$/.test(value)) found.add(value); };
  for (const m of text.matchAll(/\b(?:icon|trailingIcon|leadingIcon)\s*[:=]\s*\{?\s*['"]([a-z0-9_]+)['"]/g)) add(m[1]);
  for (const m of text.matchAll(/\bicon: string = '([a-z0-9_]+)'/g)) add(m[1]);
  for (const m of text.matchAll(/<Icon\b[^>]*?\bname=\{([^}]*)\}/g)) for (const q of m[1].matchAll(/['"]([a-z0-9_]+)['"]/g)) add(q[1]);
  for (const m of text.matchAll(/<Icon\b[^>]*?\bname="([a-z0-9_]+)"/g)) add(m[1]);
  for (const m of text.matchAll(/showToast\([^;]*?,\s*['"]([a-z0-9_]+)['"]\s*\)/g)) add(m[1]);
  for (const m of text.matchAll(/material-symbols-outlined[^>]*>\s*([a-z0-9_]+)\s*</g)) add(m[1]);
  return found;
}

test('every icon the code uses is in the bundled icon font subset', () => {
  const missing = [];
  let checked = 0;
  for (const file of files(src)) {
    for (const name of usedIcons(fs.readFileSync(file, 'utf8'))) {
      checked++;
      if (!subset.has(name)) missing.push(`${path.relative(src, file)}: ${name}`);
    }
  }
  assert.ok(checked > 40, `expected to find many icons, found ${checked}`);
  assert.deepEqual(missing, [], 'add these to public/fonts (see public/fonts/README.md)');
});

test('the app does not load fonts from a CDN', () => {
  const html = fs.readFileSync(path.resolve(__dirname, '../index.html'), 'utf8');
  const css = fs.readFileSync(path.join(src, 'index.css'), 'utf8');
  assert.ok(!/fonts\.(googleapis|gstatic)\.com/.test(html + css));
  for (const file of ['manrope-latin', 'jetbrains-mono-latin', 'material-symbols-outlined']) {
    assert.ok(fs.existsSync(path.resolve(__dirname, `../public/fonts/${file}.woff2`)), file);
  }
});
