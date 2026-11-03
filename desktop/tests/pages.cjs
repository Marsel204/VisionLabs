// Review-progress math shared by the overview, projects, quality and export pages.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const assert = require('node:assert/strict');
const { test } = require('node:test');

const code = ts.transpileModule(fs.readFileSync(path.resolve(__dirname, '../src/pages/datasetSummary.ts'), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const loaded = { exports: {} };
vm.runInNewContext(code, { module: loaded, exports: loaded.exports });
const { summarizeDataset, datasetNameFrom, firstPendingIndex } = loaded.exports;

const image = (status, annotation_count = 0) => ({ filename: 'x.jpg', width: 1, height: 1, size_bytes: 0, status, annotation_count });

test('summary counts approved, auto-labeled and untouched images separately', () => {
  const summary = summarizeDataset([image('reviewed', 3), image('ai_labeled', 2), image('unreviewed'), image('pending', 1)]);
  assert.equal(summary.total, 4);
  assert.equal(summary.approved, 1);
  assert.equal(summary.autoLabeled, 1);
  assert.equal(summary.unreviewed, 2);
  assert.equal(summary.pending, 3);
  assert.equal(summary.annotations, 6);
  assert.equal(summary.approvedPercent, 25);
});

test('an empty dataset reports zero progress instead of dividing by zero', () => {
  assert.equal(summarizeDataset([]).approvedPercent, 0);
});

test('dataset name is the last folder of the path on Linux and Windows', () => {
  assert.equal(datasetNameFrom('/home/u/Dataset/Final/'), 'Final');
  assert.equal(datasetNameFrom('C:\\data\\traffic'), 'traffic');
  assert.equal(datasetNameFrom(undefined), '');
});

test('the review queue starts at the first image a person has not approved', () => {
  assert.equal(firstPendingIndex([image('reviewed'), image('ai_labeled'), image('unreviewed')]), 1);
  assert.equal(firstPendingIndex([image('reviewed')]), 0);
});

function loadRecents(storage) {
  const recentCode=ts.transpileModule(fs.readFileSync(path.resolve(__dirname,'../src/pages/recentDatasets.ts'),'utf8'), {
    compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022},
  }).outputText;
  const module={exports:{}};
  vm.runInNewContext(recentCode,{module,exports:module.exports,localStorage:storage});
  return module.exports;
}

test('removed projects stay out of history after reload and can be reopened explicitly',()=>{
  const values=new Map();const storage={getItem:key=>values.get(key),setItem:(key,value)=>values.set(key,value)};
  const recents=loadRecents(storage);
  recents.rememberDataset('/first');recents.rememberDataset('/second');
  assert.equal(recents.forgetDataset('/first'),true);
  const reloaded=loadRecents(storage);
  assert.deepEqual([...reloaded.recentDatasets()],['/second']);
  reloaded.rememberDataset('/first');
  assert.deepEqual([...reloaded.recentDatasets()],['/first','/second']);
});

test('project removal reports an unavailable local history store',()=>{
  const recents=loadRecents({getItem:()=>null,setItem:()=>{throw Error('unavailable');}});
  assert.equal(recents.forgetDataset('/first'),false);
});
