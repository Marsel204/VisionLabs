// Pure audit-fix logic: how findings change the box list.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const assert = require('node:assert/strict');
const { test } = require('node:test');

function load() {
  const code = ts.transpileModule(fs.readFileSync(path.resolve(__dirname, '../src/audit.ts'), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(code, { module, exports: module.exports, crypto: globalThis.crypto });
  return module.exports;
}
const { applyFinding, applyOcclusionFlags, visibleFindings } = load();
const geometry = { norm_left: .1, norm_top: .2, norm_right: .3, norm_bottom: .6, x: 10, y: 20, width: 20, height: 40 };
const box = (id, extra = {}) => ({ id, class_name: 'car', class_id: 0, confidence: .9, x: 0, y: 0, width: 5, height: 5,
  norm_left: 0, norm_top: 0, norm_right: .05, norm_bottom: .05, polygon_normalized: [[0, 0], [1, 0], [1, 1]], ...extra });
const finding = (kind, extra = {}) => ({ id: 'f1', kind, severity: .5, message: '', annotation_id: null,
  class_name: null, confidence: null, occlusion: 0, likely_occluded: false, box: null, ...extra });
const classes = ['car', 'motorcycle'];

test('accepting a missed finding adds a validated, occlusion-tagged box', () => {
  const result = applyFinding(finding('possibly_missed', { class_name: 'Motorcycle', box: geometry, likely_occluded: true }), [box('a')], classes);
  assert.equal(result.length, 2);
  const added = result[1];
  assert.equal(added.class_name, 'motorcycle');
  assert.equal(added.class_id, 1);
  assert.equal(added.occluded, true);
  assert.equal(added.confidence, 1);
  assert.deepEqual([added.x, added.y, added.width, added.height], [10, 20, 20, 40]);
});

test('a missed finding for a class outside the dataset cannot be applied', () => {
  assert.equal(applyFinding(finding('possibly_missed', { class_name: 'tram', box: geometry }), [], classes), null);
});

test('misaligned fix moves the box and drops its stale polygon', () => {
  const [fixed] = applyFinding(finding('misaligned', { annotation_id: 'a', box: geometry }), [box('a')], classes);
  assert.equal(fixed.id, 'a');
  assert.equal(fixed.width, 20);
  assert.equal(fixed.polygon_normalized, null);
});

test('class mismatch changes class and id only', () => {
  const [fixed] = applyFinding(finding('class_mismatch', { annotation_id: 'a', class_name: 'motorcycle' }), [box('a')], classes);
  assert.equal(fixed.class_name, 'motorcycle');
  assert.equal(fixed.class_id, 1);
  assert.equal(fixed.width, 5);
});

test('duplicate, unsupported and odd-shape findings delete the box', () => {
  for (const kind of ['duplicate', 'unsupported', 'implausible_shape']) {
    const result = applyFinding(finding(kind, { annotation_id: 'a' }), [box('a'), box('b')], classes);
    assert.deepEqual([...result.map(b => b.id)], ['b']);
  }
  assert.equal(applyFinding(finding('duplicate', { annotation_id: 'gone' }), [box('a')], classes), null);
});

test('visibleFindings hides dismissed findings and findings about deleted boxes', () => {
  const report = { findings: [
    finding('duplicate', { id: 'x', annotation_id: 'a' }),
    finding('duplicate', { id: 'y', annotation_id: 'deleted' }),
    finding('possibly_missed', { id: 'z' }),
    finding('possibly_missed', { id: 'w' }),
  ] };
  assert.deepEqual([...visibleFindings(report, [box('a')], new Set(['w'])).map(f => f.id)], ['x', 'z']);
  assert.deepEqual([...visibleFindings(null, [], new Set())], []);
});

test('occlusion flags are only added, never cleared, and no-op returns null', () => {
  const report = { occlusion: [
    { annotation_id: 'a', fraction: .5, occluded: true, occluder_ids: [] },
    { annotation_id: 'b', fraction: 0, occluded: false, occluder_ids: [] },
  ] };
  const boxes = [box('a'), box('b', { occluded: true })];
  const result = applyOcclusionFlags(boxes, report);
  assert.deepEqual([...result.map(b => !!b.occluded)], [true, true]);
  assert.equal(applyOcclusionFlags(result, report), null);
  assert.equal(applyOcclusionFlags(boxes, null), null);
});

test('display filter cycles through presets and wraps to the original', () => {
  const { nextEnhancePreset } = load();
  assert.equal(nextEnhancePreset('none'), 'sharpen');
  assert.equal(nextEnhancePreset('rain'), 'none');
  assert.equal(nextEnhancePreset('unknown'), 'none');
});

test('audit depth maps to detector views and defaults to the flipped pass', () => {
  const { auditViews } = load();
  assert.deepEqual([...auditViews('fast')], ['original']);
  assert.deepEqual([...auditViews('thorough')], ['original', 'flip']);
  assert.deepEqual([...auditViews('max')], ['original', 'flip', 'sharpen', 'contrast']);
  assert.deepEqual([...auditViews('bogus')], ['original', 'flip']);
});
