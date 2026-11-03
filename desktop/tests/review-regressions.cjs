// Execute actual TSX handlers with controlled hooks and response ordering.
// This complements native UI testing; it does not emulate browser layout.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { webcrypto } = require('node:crypto');
const ts = require('typescript');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const root = path.resolve(__dirname, '../src');
const deferred = () => { let resolve, reject; const promise = new Promise((a,b) => { resolve=a; reject=b; }); return {resolve,reject,promise}; };
const readyModels={backend:'local',models:Object.fromEntries(
  ['grounding_dino','sam2','florence2','locate_anything','yolo:yolo11n.pt']
    .map(id=>[id,{name:id,state:'ready',error:null}]))};
const jsx = (type, props) => ({type, props:props || {}});
// Native buttons and the shared Button component are both clickable buttons.
const isBtn = n => n.type==='button' || n.type==='Button';
function loadPure(file) {
  const code=ts.transpileModule(fs.readFileSync(path.join(root,file),'utf8'), {
    compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022},
  }).outputText;
  const module={exports:{}};vm.runInNewContext(code,{module,exports:module.exports,crypto:webcrypto});return module.exports;
}
function harness(file, supplied = {}, helpers = {}) {
  const state=[], refs=[], effects=[], callbacks=[]; let cursor=0, pending=[];
  const same=(a,b)=>a && b && a.length===b.length && a.every((v,i)=>Object.is(v,b[i]));
  const React={
    useState(initial) { const i=cursor++; if (!(i in state)) state[i]=typeof initial==='function'?initial():initial;
      return [state[i], v=>{state[i]=typeof v==='function'?v(state[i]):v;}]; },
    useRef(initial) { const i=cursor++; return refs[i] || (refs[i]={current:initial}); },
    useCallback(fn,deps) { const i=cursor++; if (!callbacks[i] || !same(deps,callbacks[i].deps)) callbacks[i]={fn,deps}; return callbacks[i].fn; },
    useMemo(fn,deps) { const i=cursor++; if (!callbacks[i] || !same(deps,callbacks[i].deps)) callbacks[i]={value:fn(),deps}; return callbacks[i].value; },
    useEffect(fn,deps) { const i=cursor++; if (!effects[i] || !same(deps,effects[i].deps)) {
      const old=effects[i]; effects[i]={deps,cleanup:null}; pending.push(()=>{old?.cleanup?.(); effects[i].cleanup=fn();}); } },
  };
  const api={imageIdentity:img=>img.path||img.image_id||img.filename,
    fetchHealth:async()=>({dataset_dir:'/dataset',classes:['dog','cat']}),
    fetchImages:async()=>({images:[],total:0,directory:'/dataset'}),
    fetchAllImages:async()=>({images:[],total:0,directory:'/dataset'}),
    fetchAnnotations:async()=>({boxes:[],revision:0}),
    saveAnnotations:async(_n,boxes,revision)=>({boxes,revision:(revision||0)+1}),
    fetchModelPresets:async()=>({presets:[]}),
    prepareAutoLabelModels:async()=>readyModels,
    fetchModelPreparationStatus:async()=>readyModels,
    getImageUrl:n=>n,...supplied};
  let nextTimer=0; const intervals=new Map();
  const globals={window:{location:{search:''},addEventListener(){},removeEventListener(){}}, URLSearchParams,console,
    document:{activeElement:null},
    crypto:webcrypto,setTimeout:()=>++nextTimer,clearTimeout(){},
    setInterval(fn){const id=++nextTimer;intervals.set(id,fn);return id;},
    clearInterval(id){intervals.delete(id);}};
  // Compile a source file with the same hooks, API stubs and globals as the component under test.
  const load=source=>{
    const compiled=ts.transpileModule(fs.readFileSync(path.join(root,source),'utf8'), {
      compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,target:ts.ScriptTarget.ES2022},
    }).outputText;
    const out={};vm.runInNewContext(compiled,{...globals,exports:out,require:requireFn});return out;
  };
  function requireFn(name) {
    if (name==='react') return {...React,default:React};
    if (name==='react/jsx-runtime') return {jsx,jsxs:jsx};
    if (name.endsWith('/api')) return api;
    if (name.endsWith('/audit')) return {visibleFindings:()=>[],applyFinding:()=>null,applyOcclusionFlags:()=>null};
    // Pure helpers run for real; components stay as named placeholders.
    if (name.endsWith('/datasetSummary')) return {...loadPure('pages/datasetSummary.ts'), ...helpers.datasetSummary};
    if (name.endsWith('/format')) return loadPure('format.ts');
    if (name.endsWith('/review')) return loadPure('review.ts');
    if (name.endsWith('/useAnnotationSources')) return load('pages/useAnnotationSources.ts');
    if (name.endsWith('/recentDatasets')) return {rememberDataset(){},
      recentDatasets:supplied.recentDatasets || (()=>[]),forgetDataset:supplied.forgetDataset || (()=>true)};
    if (name.endsWith('/types')) return {getClassColor:()=>({stroke:'blue',text:'blue',bg:'black'})};
    return new Proxy({}, {get:(_t,key)=>String(key)});
  }
  const exports=load(file);
  const component=exports.default || exports[Object.keys(exports)[0]];
  return {state,refs,intervals,render(props) {cursor=0;pending=[];const tree=component(props);
    return {tree,runEffects(){pending.slice().forEach(fn=>fn());}};}};
}
function find(tree,type) {
  if (!tree || typeof tree!=='object') return null;
  if (Array.isArray(tree)) {for (const node of tree) {const result=find(node,type);if(result)return result;}return null;}
  if (tree.type===type) return tree;
  const children=tree.props?.children;
  for (const child of Array.isArray(children)?children:[children]) {const result=find(child,type);if(result)return result;}
  return null;
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const images=[{filename:'a.jpg',path:'/dataset/a.jpg',dataset_id:'/dataset',width:100,height:80},
  {filename:'b.jpg',path:'/dataset/b.jpg',dataset_id:'/dataset',width:100,height:80}];
const boxed=id=>({id,class_name:'cat',class_id:1,x:10,y:10,width:20,height:20,
  norm_left:.1,norm_top:.125,norm_right:.3,norm_bottom:.375,confidence:1});
function setup(supplied={}) { const h=harness('App.tsx',supplied);
  find(h.render().tree,'Sidebar').props.onNavigate('review');h.state[1]=images;
  h.render().runEffects();return h; }

test('dataset summary only recomputes when images change',()=>{
  const {summarizeDataset}=loadPure('pages/datasetSummary.ts');
  let calls=0;
  const h=harness('App.tsx',{}, {datasetSummary:{summarizeDataset(value){
    calls++;return summarizeDataset(value);
  }}});
  h.render(); // Initialize hooks without running effects or any network actions.
  h.state[1]=images.map(image=>({...image,status:'unreviewed',annotation_count:0}));
  let r=h.render();
  const initialCalls=calls;
  assert.equal(find(r.tree,'Sidebar').props.reviewCount,2);
  find(r.tree,'Sidebar').props.onNavigate('review');
  r=h.render();
  find(r.tree,'AnnotationCanvas').props.onSelectBox('box-1');
  h.state[2]=1; // Selecting another image keeps the image catalog unchanged.
  h.render();
  assert.equal(calls,initialCalls);
  h.state[1]=h.state[1].map((image,index)=>index===0
    ? {...image,status:'reviewed',annotation_count:1}:image);
  r=h.render();
  assert.equal(calls,initialCalls+1);
  assert.equal(find(r.tree,'Sidebar').props.reviewCount,1);
});

test('late image response cannot replace the selected document',async()=>{
  const a=deferred(),b=deferred();const h=setup({fetchAnnotations:n=>n.endsWith('a.jpg')?a.promise:b.promise});
  h.state[2]=1;h.render().runEffects();
  b.resolve({boxes:[boxed('b')],revision:2});await tick();h.render();
  a.resolve({boxes:[boxed('a')],revision:1});await tick();
  assert.equal(h.state[3][0].id,'b');
});
test('detection response after navigation never saves under the new image',async()=>{
  const result=deferred(),saved=[];const h=setup({detectYolo:()=>result.promise,
    saveAnnotations:async(n,b)=>{saved.push({n,b});return {revision:1,boxes:b};}});
  await tick();let r=h.render();const work=find(r.tree,'AnnotationCanvas').props.onRunYoloDetect();
  h.state[2]=1;h.render().runEffects();result.resolve({boxes:[boxed('detection')]});await work;
  assert.equal(saved.length,0); // Obsolete proposals require an explicit rerun on the original image.
});
test('equal filenames in different datasets trigger a new annotation request',async()=>{
  const calls=[];const h=setup({fetchAnnotations:async n=>{calls.push(n);return {boxes:[],revision:0};}});
  await tick();h.state[1]=[{...images[0],path:'/other/a.jpg',dataset_id:'/other'}];
  h.render().runEffects();await tick();assert.deepEqual(calls,['/dataset/a.jpg','/other/a.jpg']);
});
test('batch completion refreshes labels without running detection',async()=>{
  let detections=0;const h=setup({detectYolo:async()=>{detections++;return {boxes:[]};},
    fetchImages:async()=>({images,total:2,directory:'/dataset'})});
  await tick();const modal=find(h.render().tree,'AutoLabelPage');
  modal.props.onStartBatch(['cat'],.5);modal.props.onBatchComplete();await tick();
  assert.equal(detections,0);
});
test('failed saves remain visible and recoverable after navigation',async()=>{
  let failing=true;const saved=[];
  const h=setup({saveAnnotations:async(n,b)=>{if(failing)throw Error('offline');saved.push(n);return {boxes:b,revision:1};}});
  await tick();find(h.render().tree,'AnnotationCanvas').props.onAddBox(boxed('draft'));
  await tick();h.state[2]=1;h.render().runEffects();await tick();h.state[2]=0;h.render().runEffects();
  const r=h.render();assert.equal(h.state[3][0].id,'draft');
  assert.ok(find(r.tree,'div') );
  function alert(node) {if(!node || typeof node!=='object')return null;if(node.props?.role==='alert')return node;
    for(const child of [].concat(node.props?.children||[])){const found=alert(child);if(found)return found;}return null;}
  const error=alert(r.tree);assert.ok(error);failing=false;find(error,'button').props.onClick();await tick();
  assert.deepEqual(saved,['/dataset/a.jpg']);
});
test('saves for an image are serialized and carry the latest successful revision',async()=>{
  const first=deferred(),writes=[];const h=setup({saveAnnotations:async(n,b,rev)=>{writes.push({n,rev});
    if(writes.length===1)await first.promise;return {boxes:b,revision:(rev||0)+1};}});
  await tick();let canvas=find(h.render().tree,'AnnotationCanvas');canvas.props.onAddBox(boxed('first'));await tick();
  canvas=find(h.render().tree,'AnnotationCanvas');canvas.props.onAddBox(boxed('second'));await tick();
  assert.equal(writes.length,1);first.resolve();await tick();await tick();
  assert.deepEqual(writes.map(w=>w.rev),[0,1]);
});
test('drawing uses the supplied class registry, including arbitrary names',()=>{
  const added=[];const props={image:images[0],boxes:[],selectedBoxId:null,activeTool:'bbox',imageIndex:0,totalImages:2,
    activeClassName:'cat',availableClasses:['dog','cat'],onSelectBox(){},onAddBox:b=>added.push(b),isDetecting:false};
  const h=harness('components/studio/AnnotationCanvas.tsx');let r=h.render(props);
  h.refs[1].current={getBoundingClientRect:()=>({left:0,top:0,width:100,height:80})};
  r.tree.props.children[1].props.onMouseDown({button:0,buttons:1,clientX:10,clientY:10});
  r=h.render(props);r.tree.props.onMouseMove({clientX:40,clientY:40});r=h.render(props);r.tree.props.onMouseUp();
  assert.equal(added[0].class_id,1);assert.equal(added[0].source,'human');
});
test('polygon vertices are retained when the tool finishes',()=>{
  const added=[];const props={image:images[0],boxes:[],selectedBoxId:null,activeTool:'polygon',imageIndex:0,totalImages:2,
    activeClassName:'cat',availableClasses:['dog','cat'],onSelectBox(){},onAddBox:b=>added.push(b),isDetecting:false};
  const h=harness('components/studio/AnnotationCanvas.tsx');let r=h.render(props);
  h.refs[1].current={getBoundingClientRect:()=>({left:0,top:0,width:100,height:80})};
  for(const [x,y] of [[10,10],[40,10],[40,40]]) {r.tree.props.children[1].props.onMouseDown({button:0,buttons:1,clientX:x,clientY:y});r=h.render(props);}
  r.tree.props.children[1].props.onDoubleClick();assert.equal(added[0].polygon_normalized.length,3);
});
test('select drag and corner resize update normalized geometry',()=>{
  const updated=[];const b=boxed('box');const props={image:images[0],boxes:[b],selectedBoxId:'box',activeTool:'select',
    imageIndex:0,totalImages:2,availableClasses:['dog','cat'],onSelectBox(){},onAddBox(){},onUpdateBox:b=>updated.push(b),isDetecting:false};
  const h=harness('components/studio/AnnotationCanvas.tsx');let r=h.render(props);
  h.refs[1].current={getBoundingClientRect:()=>({left:0,top:0,width:100,height:80})};
  const group=find(r.tree,'g');find(group,'rect').props.onMouseDown({stopPropagation(){},clientX:15,clientY:15});
  r.tree.props.onMouseMove({clientX:25,clientY:20});r=h.render(props);r.tree.props.onMouseUp();
  assert.equal(updated[0].x,20);assert.equal(updated[0].norm_left,.2);
  r=h.render({...props,boxes:[updated[0]]});
  const handles=find(r.tree,'g').props.children.flat(Infinity).filter(node=>node?.type==='rect' && node.props.width===6);
  handles[3].props.onMouseDown({stopPropagation(){},clientX:40,clientY:35});
  r.tree.props.onMouseMove({clientX:55,clientY:50});r=h.render({...props,boxes:[updated[0]]});r.tree.props.onMouseUp();
  assert.equal(updated[1].norm_right,.55);assert.equal(updated[1].norm_bottom,.625);
});

test('detection cannot replace edits made while inference was running',async()=>{
  const result=deferred(),saved=[];const h=setup({detectYolo:()=>result.promise,
    saveAnnotations:async(n,b)=>{saved.push({n,b});return {revision:1,boxes:b};}});
  await tick();let canvas=find(h.render().tree,'AnnotationCanvas');const work=canvas.props.onRunYoloDetect();
  canvas.props.onAddBox(boxed('human'));await tick();h.render();
  result.resolve({boxes:[boxed('detection')]});await work;
  assert.equal(saved.length,1);assert.equal(saved[0].b[0].id,'human');
});

const colabStatus={supported:true,auth_state:'disconnected',auth_url:null,error:null,busy:false,job:null};
const colabProps={isOpen:true,onClose(){},health:{dataset_dir:'/dataset',classes:['car','person']},onComplete(){}};
function content(node) {
  if (node===null || node===undefined || typeof node==='boolean') return '';
  if (typeof node!=='object') return String(node);
  return [].concat(Array.isArray(node)?node:node.props?.children||[]).map(content).join('');
}
function match(tree, predicate) {
  if (!tree || typeof tree!=='object') return null;
  if (predicate(tree)) return tree;
  for (const child of [].concat(Array.isArray(tree)?tree:tree.props?.children||[])) {
    const found=match(child,predicate);if(found)return found;
  }
  return null;
}
const colabButton=(tree,label)=>match(tree,n=>isBtn(n) && content(n)===label);

test('Colab login passes a transient code and never starts annotation',async()=>{
  const calls=[];
  const h=harness('components/modals/ColabModal.tsx',{
    fetchColabStatus:async()=>colabStatus,
    connectColab:async()=>{calls.push('connect');return {...colabStatus,busy:true,auth_state:'awaiting_code',auth_url:'https://accounts.google.com/test'};},
    authorizeColab:async code=>{calls.push(code);return {...colabStatus,auth_state:'connected'};},
    startColabJob:async()=>{throw Error('Sign-in must not allocate a GPU');},
  });
  h.render(colabProps).runEffects();await tick();
  colabButton(h.render(colabProps).tree,'Masuk dengan Google').props.onClick();await tick();
  let r=h.render(colabProps);
  const input=match(r.tree,n=>n.props?.id==='colab-code');
  assert.equal(input.props.type,'password');
  input.props.onChange({target:{value:'test-one-time-code'}});
  find(h.render(colabProps).tree,'form').props.onSubmit({preventDefault(){}});await tick();
  r=h.render(colabProps);assert.deepEqual(calls,['connect','test-one-time-code']);
  assert.equal(colabButton(r.tree,'Mulai anotasi').props.disabled,false);
  assert.ok(!h.state.includes('test-one-time-code'));
});

test('Colab starts a bounded three-image mask job from the active dataset',async()=>{
  const jobs=[];
  const h=harness('components/modals/ColabModal.tsx',{
    fetchColabStatus:async()=>({...colabStatus,auth_state:'connected'}),
    startColabJob:async payload=>{jobs.push(payload);return {...colabStatus,auth_state:'connected',busy:true};},
  });
  h.render(colabProps).runEffects();await tick();
  find(h.render(colabProps).tree,'form').props.onSubmit({preventDefault(){}});await tick();
  assert.equal(jobs.length,1);
  assert.deepEqual(JSON.parse(JSON.stringify(jobs[0])),{dataset_id:'/dataset',classes:['car','person'],pipeline:'dino-masks',limit:3,gpu:'T4',confidence:.35,only_unannotated:true});
});

test('Colab exposes backend GPU choices and forwards the selected A100',async()=>{
  const jobs=[];
  const gpus=['T4','L4','A100','H100','G4'];
  const h=harness('components/modals/ColabModal.tsx',{
    fetchColabStatus:async()=>({...colabStatus,auth_state:'connected',supported_gpus:gpus}),
    startColabJob:async payload=>{jobs.push(payload);return {...colabStatus,auth_state:'connected',busy:true};},
  });
  h.render(colabProps).runEffects();await tick();
  let r=h.render(colabProps);
  const label=match(r.tree,n=>n.type==='label' && content(n).startsWith('Colab GPU'));
  const select=find(label,'select');
  assert.deepEqual(JSON.parse(JSON.stringify(select.props.children.map(option=>option.props.value || content(option)))),gpus);
  select.props.onChange({target:{value:'A100'}});
  r=h.render(colabProps);find(r.tree,'form').props.onSubmit({preventDefault(){}});await tick();
  assert.equal(jobs[0].gpu,'A100');
});

test('Colab forwards the selected Locate Anything pipeline',async()=>{
  const jobs=[];
  const h=harness('components/modals/ColabModal.tsx',{
    fetchColabStatus:async()=>({...colabStatus,auth_state:'connected'}),
    startColabJob:async payload=>{jobs.push(payload);return {...colabStatus,auth_state:'connected',busy:true};},
  });
  h.render(colabProps).runEffects();await tick();
  const select=match(h.render(colabProps).tree,n=>n.type==='select' &&
    match(n,option=>option.type==='option' && option.props.value==='locate-boxes'));
  assert.ok(select);
  select.props.onChange({target:{value:'locate-boxes'}});
  find(h.render(colabProps).tree,'form').props.onSubmit({preventDefault(){}});await tick();
  assert.equal(jobs[0].pipeline,'locate-boxes');
});

test('stale Colab status cannot undo a completed connection action',async()=>{
  const stale=deferred();let polls=0;
  const h=harness('components/modals/ColabModal.tsx',{
    fetchColabStatus:()=>++polls===1?Promise.resolve(colabStatus):stale.promise,
    connectColab:async()=>({...colabStatus,auth_state:'connected'}),
  });
  h.render(colabProps).runEffects();await tick();
  [...h.intervals.values()][0]();
  colabButton(h.render(colabProps).tree,'Masuk dengan Google').props.onClick();await tick();
  stale.resolve(colabStatus);await tick();
  assert.ok(colabButton(h.render(colabProps).tree,'Putuskan'));
});

test('a Colab job continues polling after closing and refreshes labels once',async()=>{
  let completed=0;
  const job={id:'owned-job',state:'running',stage:'Running',total:3,cleanup_required:false};
  let response={...colabStatus,auth_state:'connected',busy:true,job};
  const props={...colabProps,onComplete(){completed++;}};
  const h=harness('components/modals/ColabModal.tsx',{fetchColabStatus:async()=>response});
  h.render(props).runEffects();await tick();
  const closed={...props,isOpen:false};
  const r=h.render(closed);r.runEffects();await tick();assert.equal(r.tree,null);
  response={...response,busy:false,job:{...job,state:'completed',stage:'GPU released',saved:3}};
  [...h.intervals.values()][0]();await tick();
  h.render(closed).runEffects();assert.equal(h.intervals.size,0);assert.equal(completed,1);
  h.render(props).runEffects();await tick();assert.equal(completed,1);
});

test('Colab release failures block new work until retry or explicit manual confirmation',async()=>{
  let confirmations=0;
  const h=harness('components/modals/ColabModal.tsx',{
    fetchColabStatus:async()=>({...colabStatus,auth_state:'connected',job:{state:'cleanup_required',stage:'Release required',cleanup_required:true}}),
    acknowledgeColabRelease:async()=>{confirmations++;return {...colabStatus,auth_state:'connected'};},
  });
  h.render(colabProps).runEffects();await tick();
  const r=h.render(colabProps);
  assert.equal(find(r.tree,'fieldset').props.disabled,true);
  assert.equal(colabButton(r.tree,'Putuskan').props.disabled,true);
  assert.equal(confirmations,0);
  colabButton(r.tree,'Sudah kuhentikan di Colab').props.onClick();await tick();
  assert.equal(confirmations,1);
});

test('Colab shows the connected account and allows switching without starting a GPU',async()=>{
  let switches=0;
  const h=harness('components/modals/ColabModal.tsx',{
    fetchColabStatus:async()=>({...colabStatus,auth_state:'connected',account:{email:'free@example.com',compute_units:0,usage_rate_hourly:1.07,active_runtimes:1}}),
    switchColabAccount:async()=>{switches++;return {...colabStatus,auth_state:'connecting',busy:true};},
    startColabJob:async()=>{throw Error('Account switching must not allocate a GPU');},
  });
  h.render(colabProps).runEffects();await tick();
  const r=h.render(colabProps);
  assert.ok(content(r.tree).includes('free@example.com'));
  assert.ok(content(r.tree).includes('0.00 unit komputasi'));
  assert.ok(content(r.tree).includes('1 runtime aktif'));
  colabButton(r.tree,'Ganti akun').props.onClick();await tick();
  assert.equal(switches,1);
});

test('unsupported Colab installations offer notebook downloads without Google sign-in',async()=>{
  const downloads=[];
  const h=harness('components/modals/ColabModal.tsx',{
    fetchColabStatus:async()=>({...colabStatus,supported:false}),
    downloadColabFile:async kind=>{downloads.push(kind);},
  });
  h.render(colabProps).runEffects();await tick();
  const r=h.render(colabProps);
  assert.equal(colabButton(r.tree,'Masuk dengan Google'),null);
  colabButton(r.tree,'Unduh notebook').props.onClick();await tick();
  colabButton(h.render(colabProps).tree,'Unduh paket sumber').props.onClick();await tick();
  assert.deepEqual(downloads,['notebook','source']);
});

test('Locate Anything selection reaches preview, batch, and ensemble mode',async()=>{
  const previews=[],batches=[];
  const h=harness('pages/AutoLabelPage.tsx',{
    runAutoLabelPreview:async payload=>{previews.push(payload);return {detections:[],count:0};},
    startAutoLabelBatch:async payload=>{batches.push(payload);return {status:'started'};},
  });
  const props={isOpen:true,onClose(){},onStartBatch(){},health:{dataset_dir:'/dataset',classes:['car']},images,totalImages:2};
  let r=h.render(props);
  const checkbox=match(r.tree,n=>n.type==='input' && n.props.id==='locate-anything-detector');
  assert.equal(checkbox.props.checked,false);
  assert.ok(match(r.tree,n=>n.type==='label' && n.props.htmlFor===checkbox.props.id));
  checkbox.props.onChange({target:{checked:true}});
  h.render(props).runEffects();await tick();
  r=h.render(props);
  match(r.tree,n=>isBtn(n) && content(n).includes('Jalankan pratinjau')).props.onClick();
  await tick();
  assert.equal(previews[0].enable_locate_anything,true);
  assert.equal(previews[0].pipeline_mode,'ensemble_fusion_sam2_masks');
  // Disable the incumbent detector cards through their actual click handlers.
  for (const title of ['Grounding DINO','YOLO']) {
    const node=match(h.render(props).tree,n=>n.type==='div' && n.props.onClick && content(n).includes(title));
    node.props.onClick();
    h.render(props).runEffects();await tick();
  }
  h.render(props).runEffects();await tick();
  r=h.render(props);
  match(r.tree,n=>isBtn(n) && content(n).includes('Jalankan pratinjau')).props.onClick();
  await tick();
  assert.equal(previews.at(-1).pipeline_mode,'locate_anything_sam2_masks');
  match(h.render(props).tree,n=>isBtn(n) && content(n).includes('Mulai auto-label batch')).props.onClick();
  await tick();
  assert.equal(batches[0].enable_locate_anything,true);
  assert.equal(batches[0].enable_yolo,false);
  assert.equal(batches[0].enable_grounding_dino,false);
  assert.equal(batches[0].pipeline_mode,'locate_anything_sam2_masks');
});

const autoLabelProps={isOpen:true,onClose(){},onStartBatch(){},
  health:{dataset_dir:'/dataset',classes:['car']},images,totalImages:2};
const previewButton=tree=>match(tree,n=>isBtn(n) && content(n).includes('Jalankan pratinjau'));
const batchButton=tree=>match(tree,n=>isBtn(n) &&
  /Mulai auto-label batch|Menunggu model|Model tidak tersedia/.test(content(n)));

test('model downloads show live status and block preview and batch until ready',async()=>{
  let snapshot={...readyModels,models:{...readyModels.models,
    grounding_dino:{name:'Grounding DINO',state:'preparing',error:null},
    sam2:{name:'SAM 2',state:'queued',error:null}}};
  let previews=0,batches=0;
  const h=harness('pages/AutoLabelPage.tsx',{
    prepareAutoLabelModels:async()=>snapshot,
    fetchModelPreparationStatus:async()=>snapshot,
    runAutoLabelPreview:async()=>{previews++;return {detections:[]};},
    startAutoLabelBatch:async()=>{batches++;return {status:'started'};},
  });
  h.render(autoLabelProps).runEffects();await tick();
  let r=h.render(autoLabelProps);
  assert.ok(content(r.tree).includes('Grounding DINO: Mengunduh / memuat'));
  assert.ok(content(r.tree).includes('SAM 2: Menunggu'));
  assert.equal(previewButton(r.tree).props.disabled,true);
  assert.equal(batchButton(r.tree).props.disabled,true);
  await previewButton(r.tree).props.onClick();await batchButton(r.tree).props.onClick();
  assert.equal(previews,0);assert.equal(batches,0);
  assert.ok(!content(r.tree).includes('Validated (4/4 passed)'));

  snapshot=readyModels;
  await [...h.intervals.values()][0]();
  r=h.render(autoLabelProps);
  assert.ok(content(r.tree).includes('Model siap untuk auto-label'));
  assert.equal(previewButton(r.tree).props.disabled,false);
  assert.equal(batchButton(r.tree).props.disabled,false);
  await batchButton(r.tree).props.onClick();assert.equal(batches,1);
});

test('model download failures explain recovery and retry unlocks labeling',async()=>{
  let attempts=0;
  const failed={...readyModels,models:{...readyModels.models,
    grounding_dino:{name:'Grounding DINO',state:'error',error:'Connection interrupted'}}};
  const h=harness('pages/AutoLabelPage.tsx',{
    prepareAutoLabelModels:async()=>++attempts===1?failed:readyModels,
  });
  h.render(autoLabelProps).runEffects();await tick();
  let r=h.render(autoLabelProps);
  assert.ok(content(r.tree).includes('Connection interrupted'));
  assert.equal(batchButton(r.tree).props.disabled,true);
  colabButton(r.tree,'Coba siapkan ulang').props.onClick();
  h.render(autoLabelProps).runEffects();await tick();
  r=h.render(autoLabelProps);
  assert.equal(attempts,2);
  assert.equal(batchButton(r.tree).props.disabled,false);
  assert.ok(!content(r.tree).includes('Connection interrupted'));
});

test('warmup failure shows a warning while the loaded model stays usable',async()=>{
  const snapshot={...readyModels,models:{...readyModels.models,
    'yolo:yolo11n.pt':{name:'yolo11n.pt',state:'ready',error:null,loaded:true,
      inference_ready:false,warning:'Warmup failed; model loaded but not warmed: test failure'}}};
  const h=harness('pages/AutoLabelPage.tsx',{
    prepareAutoLabelModels:async()=>snapshot,
  });
  h.render(autoLabelProps).runEffects();await tick();
  const r=h.render(autoLabelProps);
  assert.ok(content(r.tree).includes('Dimuat; warmup belum berhasil'));
  assert.ok(content(r.tree).includes(snapshot.models['yolo:yolo11n.pt'].warning));
  assert.equal(previewButton(r.tree).props.disabled,false);
  assert.equal(batchButton(r.tree).props.disabled,false);
});

test('a stale preparation response cannot unlock a newly selected model',async()=>{
  const first=deferred(),second=deferred();const calls=[];
  const h=harness('pages/AutoLabelPage.tsx',{
    prepareAutoLabelModels:options=>{calls.push(options);return calls.length===1?first.promise:second.promise;},
  });
  h.render(autoLabelProps).runEffects();
  const r=h.render(autoLabelProps);
  match(r.tree,n=>n.type==='input' && n.props.id==='locate-anything-detector')
    .props.onChange({target:{checked:true}});
  h.render(autoLabelProps).runEffects();
  assert.equal(calls[1].enable_locate_anything,true);
  first.resolve(readyModels);await tick();
  assert.equal(batchButton(h.render(autoLabelProps).tree).props.disabled,true);
  assert.equal(h.intervals.size,0);
  second.resolve(readyModels);await tick();
  assert.equal(batchButton(h.render(autoLabelProps).tree).props.disabled,false);
  h.render({...autoLabelProps,isOpen:false}).runEffects();
  assert.equal(h.intervals.size,0);
});

test('cloud labeling stays available without local model downloads',async()=>{
  const h=harness('pages/AutoLabelPage.tsx',{
    prepareAutoLabelModels:async()=>({backend:'cloud',models:{}}),
  });
  h.render(autoLabelProps).runEffects();await tick();
  const r=h.render(autoLabelProps);
  assert.ok(content(r.tree).includes('Tidak perlu mengunduh model lokal'));
  assert.equal(previewButton(r.tree).props.disabled,false);
  assert.equal(batchButton(r.tree).props.disabled,false);
});

test('a failed preparation request shows a retry instead of waiting forever',async()=>{
  const h=harness('pages/AutoLabelPage.tsx',{
    prepareAutoLabelModels:async()=>{throw Error('API unavailable');},
  });
  h.render(autoLabelProps).runEffects();await tick();
  const r=h.render(autoLabelProps);
  assert.ok(content(r.tree).includes('Model tidak bisa disiapkan: Error: API unavailable'));
  assert.ok(colabButton(r.tree,'Coba siapkan ulang'));
  assert.equal(batchButton(r.tree).props.disabled,true);
  assert.equal(h.intervals.size,0);
});

test('labeling requires an object detector even when verifier and masks are ready',async()=>{
  const h=harness('pages/AutoLabelPage.tsx');
  h.render(autoLabelProps).runEffects();await tick();
  const tree=h.render(autoLabelProps).tree;
  for(const title of ['Grounding DINO','YOLO']) {
    const node=match(tree,n=>n.type==='div' && n.props.onClick && content(n).includes(title));
    node.props.onClick();
  }
  const r=h.render(autoLabelProps);r.runEffects();
  assert.ok(content(r.tree).includes('Pilih detektor objek'));
  assert.equal(previewButton(r.tree).props.disabled,true);
  assert.equal(batchButton(r.tree).props.disabled,true);
  assert.equal(h.intervals.size,0);
});

const classNameInput=(tree,id)=>match(tree,n=>n.type==='input' && n.props.id===`class-name-${id}`);

test('class names are editable and renamed classes reach preview and batch',async()=>{
  const previews=[],batches=[];
  const h=harness('pages/AutoLabelPage.tsx',{
    runAutoLabelPreview:async payload=>{previews.push(payload);return {detections:[]};},
    startAutoLabelBatch:async payload=>{batches.push(payload);return {status:'started'};},
  });
  h.render(autoLabelProps).runEffects();await tick();
  const input=classNameInput(h.render(autoLabelProps).tree,'01');
  assert.ok(input,'Class names must be editable');
  assert.equal(input.props.value,'Car');
  input.props.onChange({target:{value:'Traffic sign'}});
  let r=h.render(autoLabelProps);
  assert.equal(classNameInput(r.tree,'01').props.value,'Traffic sign');
  await previewButton(r.tree).props.onClick();await tick();
  assert.equal(previews[0].classes[0].name,'Traffic sign');
  assert.equal(previews[0].classes[0].prompt,'Traffic sign, visual entity');
  r=h.render(autoLabelProps);await batchButton(r.tree).props.onClick();
  assert.equal(batches[0].classes[0].name,'Traffic sign');
});

test('renaming a class preserves its custom detection prompt',()=>{
  const h=harness('pages/AutoLabelPage.tsx');
  let r=h.render(autoLabelProps);
  const input=classNameInput(r.tree,'01');
  assert.ok(input,'Class names must be editable');
  match(r.tree,n=>isBtn(n) && n.props.title==='Edit prompt text').props.onClick();
  r=h.render(autoLabelProps);
  find(r.tree,'textarea').props.onChange({target:{value:'a red octagonal stop sign beside the road'}});
  r=h.render(autoLabelProps);
  classNameInput(r.tree,'01').props.onChange({target:{value:'Stop sign'}});
  r=h.render(autoLabelProps);
  assert.equal(find(r.tree,'textarea').props.value,'a red octagonal stop sign beside the road');
});

test('new classes focus the name field and keep distinct identities after deletion',()=>{
  const h=harness('pages/AutoLabelPage.tsx');
  let r=h.render(autoLabelProps);
  match(r.tree,n=>isBtn(n) && content(n).includes('Tambah kelas')).props.onClick();
  r=h.render(autoLabelProps);
  const added=classNameInput(r.tree,'02');
  assert.ok(added,'New classes need an editable name field');
  assert.equal(added.props.autoFocus,true);
  added.props.onChange({target:{value:'Helmet'}});
  r=h.render(autoLabelProps);
  match(r.tree,n=>isBtn(n) && n.props['aria-label']==='Remove class Car').props.onClick();
  r=h.render(autoLabelProps);
  match(r.tree,n=>isBtn(n) && content(n).includes('Tambah kelas')).props.onClick();
  r=h.render(autoLabelProps);
  assert.equal(classNameInput(r.tree,'02').props.value,'Helmet');
  assert.ok(classNameInput(r.tree,'03'));
});

test('blank class names block labeling and surrounding whitespace is trimmed',async()=>{
  let previews=0,batches=0;
  const h=harness('pages/AutoLabelPage.tsx',{
    runAutoLabelPreview:async()=>{previews++;return {detections:[]};},
    startAutoLabelBatch:async()=>{batches++;return {status:'started'};},
  });
  h.render(autoLabelProps).runEffects();await tick();
  let r=h.render(autoLabelProps);
  const input=classNameInput(r.tree,'01');
  assert.ok(input,'Class names must be editable');
  input.props.onChange({target:{value:'   '}});
  r=h.render(autoLabelProps);
  assert.equal(classNameInput(r.tree,'01').props['aria-invalid'],true);
  assert.equal(previewButton(r.tree).props.disabled,true);
  assert.equal(batchButton(r.tree).props.disabled,true);
  await previewButton(r.tree).props.onClick();await batchButton(r.tree).props.onClick();
  assert.equal(previews,0);assert.equal(batches,0);
  classNameInput(r.tree,'01').props.onChange({target:{value:'  Pedestrian  '}});
  r=h.render(autoLabelProps);
  classNameInput(r.tree,'01').props.onBlur({target:{value:'  Pedestrian  '}});
  r=h.render(autoLabelProps);
  assert.equal(classNameInput(r.tree,'01').props.value,'Pedestrian');
  assert.equal(batchButton(r.tree).props.disabled,false);
});

test('the R-key detection layers a flipped pass only when +Flip is on',async()=>{
  const calls=[];
  const h=setup({detectYolo:async(...args)=>{calls.push(args);return {boxes:[]};}});
  await tick();let canvas=find(h.render().tree,'AnnotationCanvas');
  await canvas.props.onRunYoloDetect();
  assert.equal(calls.at(-1)[4],undefined);
  canvas.props.onChangeFlipDetect(true);
  canvas=find(h.render().tree,'AnnotationCanvas');
  assert.equal(canvas.props.flipDetect,true);
  await canvas.props.onRunYoloDetect();
  assert.deepEqual([...calls.at(-1)[4]],['original','flip']);
});

test('the Auto-Label flip pass reaches preview and batch payloads',async()=>{
  const previews=[],batches=[];
  const h=harness('pages/AutoLabelPage.tsx',{
    runAutoLabelPreview:async payload=>{previews.push(payload);return {detections:[],count:0};},
    startAutoLabelBatch:async payload=>{batches.push(payload);return {status:'started'};},
  });
  const props={isOpen:true,onClose(){},onStartBatch(){},health:{dataset_dir:'/dataset',classes:['car']},images,totalImages:2};
  h.render(props).runEffects();await tick();h.render(props).runEffects();await tick();
  let r=h.render(props);
  const flip=match(r.tree,n=>n.type==='input' && n.props.id==='yolo-flip-pass');
  assert.equal(flip.props.checked,false);
  flip.props.onChange({target:{checked:true}});
  h.render(props).runEffects();await tick();
  match(h.render(props).tree,n=>isBtn(n) && content(n).includes('Jalankan pratinjau')).props.onClick();
  await tick();
  assert.deepEqual([...previews.at(-1).yolo_views],['original','flip']);
  match(h.render(props).tree,n=>isBtn(n) && content(n).includes('Mulai auto-label batch')).props.onClick();
  await tick();
  assert.deepEqual([...batches.at(-1).yolo_views],['original','flip']);
});

// ---- review state: editing is not approving -------------------------------------------------
const approveButton=tree=>match(tree,n=>isBtn(n) && content(n).includes('Setujui gambar'));

test('editing a box saves it as unreviewed instead of approving the image',async()=>{
  const statuses=[];
  const h=setup({saveAnnotations:async(_n,b,rev,_d,status)=>{statuses.push(status);return {boxes:b,revision:(rev||0)+1,review_status:status};}});
  await tick();find(h.render().tree,'AnnotationCanvas').props.onAddBox(boxed('x'));await tick();
  assert.deepEqual(statuses,['unreviewed']);
  assert.equal(h.state[1][0].status,'unreviewed');
});

test('an explicit approval saves as reviewed and edits keep an approved image approved',async()=>{
  const statuses=[],first=deferred();
  const h=setup({saveAnnotations:async(_n,b,rev,_d,status)=>{statuses.push(status);if(statuses.length===1)await first.promise;
    return {boxes:b,revision:(rev||0)+1,review_status:status};}});
  await tick();
  const approving=approveButton(h.render().tree).props.onClick();
  // An edit made while the approval is still being written must not undo it.
  find(h.render().tree,'AnnotationCanvas').props.onAddBox(boxed('later'));
  first.resolve();await approving;await tick();await tick();
  assert.deepEqual(statuses,['reviewed','reviewed']);
  assert.equal(h.state[1][0].status,'reviewed');
});

test('detection adds new boxes and keeps the existing labels',async()=>{
  const saved=[];
  const existing={...boxed('human'),class_name:'dog',class_id:0};
  const duplicate={...boxed('server-1'),class_name:'dog',class_id:0};
  const fresh={...boxed('server-2'),class_name:'cat',norm_left:.6,norm_top:.5,norm_right:.8,norm_bottom:.8,confidence:.7,source:'yolo'};
  const h=setup({fetchAnnotations:async()=>({boxes:[existing],revision:0}),
    detectYolo:async()=>({boxes:[duplicate,fresh]}),
    saveAnnotations:async(_n,b,rev,_d,status)=>{saved.push({b,status});return {boxes:b,revision:(rev||0)+1,review_status:status};}});
  await tick();
  await find(h.render().tree,'AnnotationCanvas').props.onRunYoloDetect();
  assert.equal(saved.length,1);
  assert.equal(saved[0].b.length,2);
  assert.equal(saved[0].b[0].id,'human','the human label is untouched and first');
  assert.notEqual(saved[0].b[1].id,'server-2','new boxes get fresh ids so they cannot collide');
  assert.equal(saved[0].b[1].class_name,'cat');
  assert.equal(saved[0].status,'unreviewed','detecting does not approve the image');
});

test('a detection with nothing new does not rewrite the labels',async()=>{
  let writes=0;
  const existing={...boxed('human'),class_name:'dog',class_id:0};
  const h=setup({fetchAnnotations:async()=>({boxes:[existing],revision:0}),
    detectYolo:async()=>({boxes:[{...existing,id:'server-1'}]}),
    saveAnnotations:async(_n,b,rev)=>{writes++;return {boxes:b,revision:(rev||0)+1};}});
  await tick();
  await find(h.render().tree,'AnnotationCanvas').props.onRunYoloDetect();
  assert.equal(writes,0);
});

// ---- importing existing annotations ----------------------------------------------------------
const cocoSources=[
  {path:'/d/train/annotations.json',relative:'train/annotations.json',image_dir:'train/images',images:3,found_images:3,annotations:30,categories:['car','motorcycle']},
  {path:'/d/val/annotations.json',relative:'val/annotations.json',image_dir:'val/images',images:2,found_images:1,annotations:20,categories:['car']}];
const importProps=(over={})=>({datasetKey:'/d',onImported(){},onOpenReview(){},...over});

function importHarness(statusRef,started,extra={}) {
  return harness('pages/AnnotationImportCard.tsx',{
    fetchAnnotationSources:async()=>({directory:'/d',sources:cocoSources,skipped:[]}),
    fetchAnnotationImportStatus:async()=>statusRef.value,
    startAnnotationImport:async payload=>{started.push(payload);
      statusRef.value={running:true,completed:false,error:null,job_id:'j1',dataset_id:'/d',total:5,processed:2,imported:2,boxes:20};
      return {status:'started',job_id:'j1',files:2};},
    ...extra});
}
const startButton=tree=>match(tree,n=>isBtn(n) && /^Impor \d/.test(content(n)));

test('the import card lists the COCO files, warns about missing images, and imports every file at once',async()=>{
  const started=[],status={value:{running:false,completed:false,error:null}};
  const h=importHarness(status,started),props=importProps();
  h.render(props).runEffects();await tick();await tick();
  const r=h.render(props);const text=content(r.tree);
  assert.ok(text.includes('train/annotations.json')&&text.includes('val/annotations.json'));
  assert.ok(text.includes('1 gambar tidak ditemukan di val/images/'));
  const button=startButton(r.tree);
  assert.equal(content(button),'Impor 50 kotak dari 5 gambar');
  await button.props.onClick();
  assert.deepEqual(JSON.parse(JSON.stringify(started)),[{sources:['/d/train/annotations.json','/d/val/annotations.json'],overwrite:false,dataset_id:'/d'}]);
});

test('overwriting existing labels is off by default and sent only when ticked',async()=>{
  const started=[],status={value:{running:false,completed:false,error:null}};
  const h=importHarness(status,started),props=importProps();
  h.render(props).runEffects();await tick();await tick();
  const box=match(h.render(props).tree,n=>n.type==='input' && n.props.type==='checkbox');
  assert.equal(box.props.checked,false);
  box.props.onChange({target:{checked:true}});
  await startButton(h.render(props).tree).props.onClick();
  assert.equal(started[0].overwrite,true);
});

test('progress is shown while importing and the lists refresh exactly once when it finishes',async()=>{
  const started=[],status={value:{running:false,completed:false,error:null}},refreshed=[];
  const h=importHarness(status,started),props=importProps({onImported:()=>refreshed.push(1)});
  h.render(props).runEffects();await tick();await tick();
  await startButton(h.render(props).tree).props.onClick();await tick();await tick();
  h.render(props).runEffects();await tick();
  let text=content(h.render(props).tree);
  assert.ok(text.includes('Mengimpor… 40%'),'2 of 5 images is 40%');
  assert.equal(refreshed.length,0);
  status.value={running:false,completed:true,error:null,job_id:'j1',dataset_id:'/d',total:5,processed:5,imported:4,boxes:48,missing_images:1,classes_added:['tuktuk']};
  await [...h.intervals.values()][0]();await tick();
  h.render(props).runEffects();await tick();
  text=content(h.render(props).tree);
  assert.ok(text.includes('Impor selesai'));
  assert.ok(text.includes('1 gambar tidak ditemukan di disk'));
  assert.ok(text.includes('Kelas baru ditambahkan: tuktuk'));
  assert.ok(text.includes('perlu review'),'imported images still need approval');
  h.render(props).runEffects();await tick();h.render(props).runEffects();
  assert.equal(refreshed.length,1);
});

test('a COCO file elsewhere can be imported by path, and start errors are shown',async()=>{
  const started=[],status={value:{running:false,completed:false,error:null}};
  const h=importHarness(status,started,{startAnnotationImport:async payload=>{started.push(payload);throw Error('422: Annotation file not found: /nope.json');}});
  const props=importProps();h.render(props).runEffects();await tick();await tick();
  match(h.render(props).tree,n=>n.type==='input' && n.props.id==='coco-path').props.onChange({target:{value:'  /nope.json '}});
  await match(h.render(props).tree,n=>isBtn(n) && content(n)==='Impor file ini').props.onClick();await tick();await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(started[0].sources)),['/nope.json']);
  assert.ok(content(h.render(props).tree).includes('Annotation file not found'));
});

test('the card says so when the folder has no COCO file',async()=>{
  const h=harness('pages/AnnotationImportCard.tsx',{fetchAnnotationSources:async()=>({directory:'/d',sources:[],skipped:[]}),
    fetchAnnotationImportStatus:async()=>({running:false,completed:false,error:null})});
  const props=importProps();h.render(props).runEffects();await tick();await tick();
  const r=h.render(props);
  assert.ok(content(r.tree).includes('Tidak ada file anotasi COCO di folder ini'));
  assert.equal(startButton(r.tree),null);
});

const projectProps={images:[],health:{dataset_dir:'/dataset',classes:[]},datasetName:'dataset',
  onNavigate(){},onDatasetChanged(){},onCloseProject:async()=>{}};
const removeProjectButton=(tree,name)=>match(tree,n=>isBtn(n) && n.props['aria-label']===`Hapus proyek ${name}`);
const confirmProjectButton=tree=>match(tree,n=>isBtn(n) && content(n)==='Hapus proyek' && !n.props['aria-label']);

test('project removal requires confirmation and cancel keeps every entry',()=>{
  let forgotten=0,closed=0;
  const h=harness('pages/ProjectsPage.tsx',{recentDatasets:()=>['/dataset','/older'],forgetDataset:()=>{forgotten++;return true;}});
  const props={...projectProps,onCloseProject:async()=>{closed++;}};
  removeProjectButton(h.render(props).tree,'dataset').props.onClick();
  assert.ok(content(h.render(props).tree).includes('File gambar dan anotasi tetap tersimpan'));
  match(h.render(props).tree,n=>isBtn(n) && content(n)==='Batal').props.onClick();
  assert.ok(removeProjectButton(h.render(props).tree,'dataset'));
  assert.equal(forgotten,0);assert.equal(closed,0);
});

test('removing a recent project keeps the active project open',async()=>{
  let recent=['/dataset','/older'],closed=0;
  const h=harness('pages/ProjectsPage.tsx',{recentDatasets:()=>recent,
    forgetDataset:path=>{recent=recent.filter(item=>item!==path);return true;}});
  const props={...projectProps,onCloseProject:async()=>{closed++;}};
  removeProjectButton(h.render(props).tree,'older').props.onClick();
  confirmProjectButton(h.render(props).tree).props.onClick();await tick();
  assert.deepEqual(recent,['/dataset']);assert.equal(closed,0);
  assert.equal(removeProjectButton(h.render(props).tree,'older'),null);
});

test('failed project close keeps its entry and lets the user retry',async()=>{
  let forgotten=0;
  const h=harness('pages/ProjectsPage.tsx',{recentDatasets:()=>['/dataset'],forgetDataset:()=>{forgotten++;return true;}});
  const props={...projectProps,onCloseProject:async()=>{throw Error('Tunggu auto-label selesai');}};
  removeProjectButton(h.render(props).tree,'dataset').props.onClick();
  confirmProjectButton(h.render(props).tree).props.onClick();await tick();
  assert.equal(forgotten,0);
  assert.ok(content(h.render(props).tree).includes('Tunggu auto-label selesai'));
  assert.equal(confirmProjectButton(h.render(props).tree).props.disabled,false);
});

test('removing the active project waits for saves and refreshes to an empty workspace',async()=>{
  const save=deferred();let closed=false;
  const h=setup({saveAnnotations:()=>save.promise,closeProject:async()=>{closed=true;},
    fetchHealth:async()=>({dataset_dir:closed?'':'/dataset',classes:[]}),
    fetchImages:async()=>({images:closed?[]:images,total:closed?0:2,directory:closed?'':'/dataset'})});
  await tick();find(h.render().tree,'AnnotationCanvas').props.onAddBox(boxed('saved'));
  find(h.render().tree,'Sidebar').props.onNavigate('projects');
  const work=find(h.render().tree,'ProjectsPage').props.onCloseProject('/dataset');
  await tick();assert.equal(closed,false);
  save.resolve({revision:1,boxes:[boxed('saved')]});await work;await tick();
  const projects=find(h.render().tree,'ProjectsPage');
  assert.equal(closed,true);assert.equal(projects.props.health.dataset_dir,'');assert.equal(projects.props.images.length,0);
});

test('unsaved annotations block closing the active project',async()=>{
  let closed=false;
  const h=setup({saveAnnotations:async()=>{throw Error('disk full');},closeProject:async()=>{closed=true;}});
  await tick();find(h.render().tree,'AnnotationCanvas').props.onAddBox(boxed('unsaved'));await tick();
  find(h.render().tree,'Sidebar').props.onNavigate('projects');
  await assert.rejects(find(h.render().tree,'ProjectsPage').props.onCloseProject('/dataset'),/belum tersimpan/);
  assert.equal(closed,false);
});

const classProps={classes:['motorcycle','car','bus'],datasetId:'/dataset',onManage:async()=>({classes:[],backup:'.visionlab/class-backups/test'})};
const classButton=(tree,label)=>match(tree,node=>isBtn(node) && node.props['aria-label']===label);
const submitClasses=async tree=>{find(tree,'form').props.onSubmit({preventDefault(){}});await tick();};

test('class rename requires confirmation, sends the registry snapshot and supports cancel',async()=>{
  const changes=[];
  const h=harness('pages/ClassManager.tsx');
  const props={...classProps,onManage:async change=>{changes.push(change);return {classes:['motorbike','car','bus'],backup:'backup/test'};}};
  classButton(h.render(props).tree,'Ubah nama kelas motorcycle').props.onClick();
  assert.equal(changes.length,0);
  match(h.render(props).tree,node=>isBtn(node) && content(node)==='Batal').props.onClick();
  assert.equal(find(h.render(props).tree,'form'),null);
  classButton(h.render(props).tree,'Ubah nama kelas motorcycle').props.onClick();
  find(h.render(props).tree,'input').props.onChange({target:{value:'motorbike'}});
  await submitClasses(h.render(props).tree);
  assert.equal(changes[0].target,'motorbike');
  assert.equal(changes[0].dataset_id,'/dataset');
  assert.deepEqual([...changes[0].expected_classes],classProps.classes);
  assert.ok(content(h.render(props).tree).includes('backup/test'));
});

test('deleting a class defaults to unused-only and requires explicit annotation deletion',async()=>{
  const changes=[];
  const h=harness('pages/ClassManager.tsx');
  const props={...classProps,onManage:async change=>{changes.push(change);return {classes:[],backup:'backup/test'};}};
  classButton(h.render(props).tree,'Hapus kelas motorcycle').props.onClick();
  assert.equal(changes.length,0);
  await submitClasses(h.render(props).tree);
  assert.equal(changes[0].delete_annotations,false);
  assert.equal(changes[0].target,undefined);
  classButton(h.render(props).tree,'Hapus kelas motorcycle').props.onClick();
  find(h.render(props).tree,'select').props.onChange({target:{value:'delete'}});
  assert.ok(content(h.render(props).tree).includes('Semua box dan poligon kelas ini akan dihapus'));
  await submitClasses(h.render(props).tree);
  assert.equal(changes[1].delete_annotations,true);
});

test('class deletion can move annotations and errors remain recoverable',async()=>{
  const changes=[];
  const h=harness('pages/ClassManager.tsx');
  const props={...classProps,onManage:async change=>{changes.push(change);throw Error('disk full');}};
  classButton(h.render(props).tree,'Hapus kelas motorcycle').props.onClick();
  find(h.render(props).tree,'select').props.onChange({target:{value:'move:car'}});
  await submitClasses(h.render(props).tree);
  assert.equal(changes[0].target,'car');
  assert.equal(changes[0].delete_annotations,false);
  assert.ok(content(h.render(props).tree).includes('disk full'));
  assert.equal(match(h.render(props).tree,node=>isBtn(node) && node.props.type==='submit').props.disabled,false);
});

test('class rename disables empty and duplicate names',()=>{
  const h=harness('pages/ClassManager.tsx');
  classButton(h.render(classProps).tree,'Ubah nama kelas motorcycle').props.onClick();
  for(const value of ['', 'CAR']) {
    find(h.render(classProps).tree,'input').props.onChange({target:{value}});
    assert.equal(match(h.render(classProps).tree,node=>isBtn(node) && node.props.type==='submit').props.disabled,true);
  }
});

test('class management blocks failed annotation saves',async()=>{
  let changed=false;
  const h=setup({saveAnnotations:async()=>{throw Error('disk full');},changeDatasetClass:async()=>{changed=true;}});
  await tick();find(h.render().tree,'AnnotationCanvas').props.onAddBox(boxed('unsaved'));await tick();
  find(h.render().tree,'Sidebar').props.onNavigate('import');
  await assert.rejects(find(h.render().tree,'ImportPage').props.onManageClasses({dataset_id:'/dataset'}),/belum tersimpan/);
  assert.equal(changed,false);
});

test('class management waits for pending saves before migrating annotations',async()=>{
  const save=deferred();let changed=false;
  const h=setup({saveAnnotations:()=>save.promise,changeDatasetClass:async()=>{changed=true;return {classes:['cat'],backup:'backup/test'};}});
  await tick();find(h.render().tree,'AnnotationCanvas').props.onAddBox(boxed('saved'));
  find(h.render().tree,'Sidebar').props.onNavigate('import');
  const work=find(h.render().tree,'ImportPage').props.onManageClasses({dataset_id:'/dataset'});
  await tick();assert.equal(changed,false);
  save.resolve({revision:1,boxes:[boxed('saved')]});await work;
  assert.equal(changed,true);
});
