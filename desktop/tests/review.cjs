// Review logic: merging detections without losing labels, review status, and keyboard focus rules.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const assert = require('node:assert/strict');
const { test } = require('node:test');

const code = ts.transpileModule(fs.readFileSync(path.resolve(__dirname, '../src/review.ts'), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const loaded = { exports: {} };
vm.runInNewContext(code, { module: loaded, exports: loaded.exports, crypto: globalThis.crypto });
const { statusAfterEdit, isAlreadyLabelled, mergeDetections, isTypingTarget, activatesFocusedControl } = loaded.exports;

const box = (id, cls, l, t, r, b, extra = {}) => ({ id, class_name: cls, class_id: 0, confidence: 1,
  x: 0, y: 0, width: 0, height: 0, norm_left: l, norm_top: t, norm_right: r, norm_bottom: b, ...extra });

test('editing keeps the review state instead of approving the image', () => {
  assert.equal(statusAfterEdit('reviewed'), 'reviewed');
  assert.equal(statusAfterEdit('ai_labeled'), 'ai_labeled');
  assert.equal(statusAfterEdit('unreviewed'), 'unreviewed');
  assert.equal(statusAfterEdit(undefined), 'unreviewed');
  assert.equal(statusAfterEdit('pending'), 'unreviewed');
});

test('a detection that repeats an existing label is not added again', () => {
  const existing = [box('a', 'motorcycle', .10, .10, .30, .40)];
  assert.ok(isAlreadyLabelled(box('d', 'motorcycle', .11, .10, .31, .41), existing));
  assert.ok(isAlreadyLabelled(box('d', 'car', .10, .10, .30, .40), existing), 'the human label wins over a different class');
  assert.ok(!isAlreadyLabelled(box('d', 'motorcycle', .60, .50, .80, .90), existing));
});

test('a small box inside a larger one is a hidden neighbour, not a duplicate', () => {
  const existing = [box('car', 'car', .10, .10, .70, .80)];
  assert.ok(!isAlreadyLabelled(box('d', 'motorcycle', .20, .30, .32, .55), existing));
  assert.ok(!isAlreadyLabelled(box('d', 'car', .20, .30, .32, .55), existing), 'same class but a much smaller object');
});

test('merging keeps every existing box untouched and gives new boxes fresh ids', () => {
  const existing = [box('yolo-1', 'car', .1, .1, .3, .3), box('yolo-2', 'car', .5, .5, .7, .7)];
  const detections = [
    box('yolo-1', 'car', .1, .1, .3, .3),        // repeats an existing box (and reuses its id)
    box('yolo-2', 'bus', .0, .6, .2, .9),        // new object that happens to reuse an id
  ];
  let n = 0;
  const merged = mergeDetections(existing, detections, () => `new-${++n}`);
  assert.deepEqual([...merged.boxes.map(b => b.id)], ['yolo-1', 'yolo-2', 'new-1']);
  assert.equal(merged.added, 1);
  assert.equal(merged.skipped, 1);
  assert.equal(merged.boxes[0], existing[0]);
  assert.equal(merged.boxes[2].class_name, 'bus');
});

test('detections that overlap each other are collapsed as they are added', () => {
  const merged = mergeDetections([], [box('1', 'car', .1, .1, .3, .3), box('2', 'car', .11, .1, .31, .3)]);
  assert.equal(merged.added, 1);
});

test('shortcuts stay off while typing or choosing, but not after ticking a checkbox', () => {
  const el = (tagName, extra = {}) => ({ tagName, ...extra });
  assert.ok(isTypingTarget(el('TEXTAREA')));
  assert.ok(isTypingTarget(el('SELECT')));
  assert.ok(isTypingTarget(el('INPUT', { type: 'text' })));
  assert.ok(isTypingTarget(el('INPUT', { type: 'number' })));
  assert.ok(isTypingTarget(el('INPUT', { type: 'range' })));
  assert.ok(isTypingTarget(el('DIV', { isContentEditable: true })));
  assert.ok(!isTypingTarget(el('INPUT', { type: 'checkbox' })));
  assert.ok(!isTypingTarget(el('INPUT', { type: 'radio' })));
  assert.ok(!isTypingTarget(el('BUTTON')));
  assert.ok(!isTypingTarget(el('BODY')));
  assert.ok(!isTypingTarget(null));
});

test('Enter and Space belong to a focused control, other keys never do', () => {
  const el = (tagName, extra = {}) => ({ tagName, getAttribute: () => null, ...extra });
  assert.ok(activatesFocusedControl(el('BUTTON'), 'Enter'));
  assert.ok(activatesFocusedControl(el('BUTTON'), ' '));
  assert.ok(activatesFocusedControl(el('A'), 'Enter'));
  assert.ok(activatesFocusedControl(el('INPUT', { type: 'checkbox' }), ' '));
  assert.ok(activatesFocusedControl(el('DIV', { getAttribute: name => name === 'role' ? 'button' : null }), 'Enter'));
  assert.ok(!activatesFocusedControl(el('BUTTON'), 'Delete'));
  assert.ok(!activatesFocusedControl(el('BUTTON'), 'd'));
  assert.ok(!activatesFocusedControl(el('BODY'), 'Enter'));
});
