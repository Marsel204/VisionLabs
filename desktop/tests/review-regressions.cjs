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
function harness(file, supplied = {}) {
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
  const code=ts.transpileModule(fs.readFileSync(path.join(root,file),'utf8'), {
    compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,target:ts.ScriptTarget.ES2022},
  }).outputText;
  const exports={}; let nextTimer=0; const intervals=new Map();
  vm.runInNewContext(code, {exports, require(name) {
    if (name==='react') return {...React,default:React};
    if (name==='react/jsx-runtime') return {jsx,jsxs:jsx};
    if (name.endsWith('/api')) return api;
    if (name.endsWith('/types')) return {getClassColor:()=>({stroke:'blue',text:'blue',bg:'black'})};
    return new Proxy({}, {get:(_t,key)=>String(key)});
  }, window:{location:{search:''},addEventListener(){},removeEventListener(){}}, URLSearchParams,console,
  document:{activeElement:null},
  crypto:webcrypto,setTimeout:()=>++nextTimer,clearTimeout(){},
  setInterval(fn){const id=++nextTimer;intervals.set(id,fn);return id;},
  clearInterval(id){intervals.delete(id);} });
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
function setup(supplied={}) { const h=harness('App.tsx',supplied);h.render();h.state[1]=images;
  h.render().runEffects();return h; }

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
  await tick();const modal=find(h.render().tree,'AutoLabelModal');
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
const colabButton=(tree,label)=>match(tree,n=>n.type==='button' && content(n)===label);

test('Colab login passes a transient code and never starts annotation',async()=>{
  const calls=[];
  const h=harness('components/modals/ColabModal.tsx',{
    fetchColabStatus:async()=>colabStatus,
    connectColab:async()=>{calls.push('connect');return {...colabStatus,busy:true,auth_state:'awaiting_code',auth_url:'https://accounts.google.com/test'};},
    authorizeColab:async code=>{calls.push(code);return {...colabStatus,auth_state:'connected'};},
    startColabJob:async()=>{throw Error('Sign-in must not allocate a GPU');},
  });
  h.render(colabProps).runEffects();await tick();
  colabButton(h.render(colabProps).tree,'Sign in with Google').props.onClick();await tick();
  let r=h.render(colabProps);
  const input=match(r.tree,n=>n.props?.id==='colab-code');
  assert.equal(input.props.type,'password');
  input.props.onChange({target:{value:'test-one-time-code'}});
  find(h.render(colabProps).tree,'form').props.onSubmit({preventDefault(){}});await tick();
  r=h.render(colabProps);assert.deepEqual(calls,['connect','test-one-time-code']);
  assert.equal(colabButton(r.tree,'Start annotation').props.disabled,false);
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
  colabButton(h.render(colabProps).tree,'Sign in with Google').props.onClick();await tick();
  stale.resolve(colabStatus);await tick();
  assert.ok(colabButton(h.render(colabProps).tree,'Disconnect'));
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
  assert.equal(colabButton(r.tree,'Disconnect').props.disabled,true);
  assert.equal(confirmations,0);
  colabButton(r.tree,"I've stopped it in Colab").props.onClick();await tick();
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
  assert.ok(content(r.tree).includes('0.00 compute units'));
  assert.ok(content(r.tree).includes('1 active runtime'));
  colabButton(r.tree,'Switch account').props.onClick();await tick();
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
  assert.equal(colabButton(r.tree,'Sign in with Google'),null);
  colabButton(r.tree,'Download notebook').props.onClick();await tick();
  colabButton(h.render(colabProps).tree,'Download source bundle').props.onClick();await tick();
  assert.deepEqual(downloads,['notebook','source']);
});

test('Locate Anything selection reaches preview, batch, and ensemble mode',async()=>{
  const previews=[],batches=[];
  const h=harness('components/modals/AutoLabelModal.tsx',{
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
  match(r.tree,n=>n.type==='button' && content(n).includes('Run Test Preview')).props.onClick();
  await tick();
  assert.equal(previews[0].enable_locate_anything,true);
  assert.equal(previews[0].pipeline_mode,'ensemble_fusion_sam2_masks');
  // Disable the incumbent detector cards through their actual click handlers.
  for (const title of ['Grounding DINO 1.5 Pro','YOLO & Custom Weights Ensemble']) {
    const node=match(h.render(props).tree,n=>n.type==='div' && n.props.onClick && content(n).includes(title));
    node.props.onClick();
    h.render(props).runEffects();await tick();
  }
  h.render(props).runEffects();await tick();
  r=h.render(props);
  match(r.tree,n=>n.type==='button' && content(n).includes('Run Test Preview')).props.onClick();
  await tick();
  assert.equal(previews.at(-1).pipeline_mode,'locate_anything_sam2_masks');
  match(h.render(props).tree,n=>n.type==='button' && content(n).includes('Start Batch Auto-Labeling')).props.onClick();
  await tick();
  assert.equal(batches[0].enable_locate_anything,true);
  assert.equal(batches[0].enable_yolo,false);
  assert.equal(batches[0].enable_grounding_dino,false);
  assert.equal(batches[0].pipeline_mode,'locate_anything_sam2_masks');
});

const autoLabelProps={isOpen:true,onClose(){},onStartBatch(){},
  health:{dataset_dir:'/dataset',classes:['car']},images,totalImages:2};
const previewButton=tree=>match(tree,n=>n.type==='button' && content(n).includes('Run Test Preview'));
const batchButton=tree=>match(tree,n=>n.type==='button' &&
  /Start Batch Auto-Labeling|Waiting for models|Models unavailable/.test(content(n)));

test('model downloads show live status and block preview and batch until ready',async()=>{
  let snapshot={...readyModels,models:{...readyModels.models,
    grounding_dino:{name:'Grounding DINO',state:'preparing',error:null},
    sam2:{name:'SAM 2',state:'queued',error:null}}};
  let previews=0,batches=0;
  const h=harness('components/modals/AutoLabelModal.tsx',{
    prepareAutoLabelModels:async()=>snapshot,
    fetchModelPreparationStatus:async()=>snapshot,
    runAutoLabelPreview:async()=>{previews++;return {detections:[]};},
    startAutoLabelBatch:async()=>{batches++;return {status:'started'};},
  });
  h.render(autoLabelProps).runEffects();await tick();
  let r=h.render(autoLabelProps);
  assert.ok(content(r.tree).includes('Grounding DINO: Downloading / loading'));
  assert.ok(content(r.tree).includes('SAM 2: Waiting'));
  assert.equal(previewButton(r.tree).props.disabled,true);
  assert.equal(batchButton(r.tree).props.disabled,true);
  await previewButton(r.tree).props.onClick();await batchButton(r.tree).props.onClick();
  assert.equal(previews,0);assert.equal(batches,0);
  assert.ok(!content(r.tree).includes('Validated (4/4 passed)'));

  snapshot=readyModels;
  await [...h.intervals.values()][0]();
  r=h.render(autoLabelProps);
  assert.ok(content(r.tree).includes('Models ready for Auto Labeling'));
  assert.equal(previewButton(r.tree).props.disabled,false);
  assert.equal(batchButton(r.tree).props.disabled,false);
  await batchButton(r.tree).props.onClick();assert.equal(batches,1);
});

test('model download failures explain recovery and retry unlocks labeling',async()=>{
  let attempts=0;
  const failed={...readyModels,models:{...readyModels.models,
    grounding_dino:{name:'Grounding DINO',state:'error',error:'Connection interrupted'}}};
  const h=harness('components/modals/AutoLabelModal.tsx',{
    prepareAutoLabelModels:async()=>++attempts===1?failed:readyModels,
  });
  h.render(autoLabelProps).runEffects();await tick();
  let r=h.render(autoLabelProps);
  assert.ok(content(r.tree).includes('Connection interrupted'));
  assert.equal(batchButton(r.tree).props.disabled,true);
  colabButton(r.tree,'Retry model preparation').props.onClick();
  h.render(autoLabelProps).runEffects();await tick();
  r=h.render(autoLabelProps);
  assert.equal(attempts,2);
  assert.equal(batchButton(r.tree).props.disabled,false);
  assert.ok(!content(r.tree).includes('Connection interrupted'));
});

test('a stale preparation response cannot unlock a newly selected model',async()=>{
  const first=deferred(),second=deferred();const calls=[];
  const h=harness('components/modals/AutoLabelModal.tsx',{
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
  const h=harness('components/modals/AutoLabelModal.tsx',{
    prepareAutoLabelModels:async()=>({backend:'cloud',models:{}}),
  });
  h.render(autoLabelProps).runEffects();await tick();
  const r=h.render(autoLabelProps);
  assert.ok(content(r.tree).includes('No local model downloads are needed'));
  assert.equal(previewButton(r.tree).props.disabled,false);
  assert.equal(batchButton(r.tree).props.disabled,false);
});

test('a failed preparation request shows a retry instead of waiting forever',async()=>{
  const h=harness('components/modals/AutoLabelModal.tsx',{
    prepareAutoLabelModels:async()=>{throw Error('API unavailable');},
  });
  h.render(autoLabelProps).runEffects();await tick();
  const r=h.render(autoLabelProps);
  assert.ok(content(r.tree).includes('Cannot prepare models: Error: API unavailable'));
  assert.ok(colabButton(r.tree,'Retry model preparation'));
  assert.equal(batchButton(r.tree).props.disabled,true);
  assert.equal(h.intervals.size,0);
});

test('labeling requires an object detector even when verifier and masks are ready',async()=>{
  const h=harness('components/modals/AutoLabelModal.tsx');
  h.render(autoLabelProps).runEffects();await tick();
  const tree=h.render(autoLabelProps).tree;
  for(const title of ['Grounding DINO 1.5 Pro','YOLO & Custom Weights Ensemble']) {
    const node=match(tree,n=>n.type==='div' && n.props.onClick && content(n).includes(title));
    node.props.onClick();
  }
  const r=h.render(autoLabelProps);r.runEffects();
  assert.ok(content(r.tree).includes('Select an object detector'));
  assert.equal(previewButton(r.tree).props.disabled,true);
  assert.equal(batchButton(r.tree).props.disabled,true);
  assert.equal(h.intervals.size,0);
});

const classNameInput=(tree,id)=>match(tree,n=>n.type==='input' && n.props.id===`class-name-${id}`);

test('class names are editable and renamed classes reach preview and batch',async()=>{
  const previews=[],batches=[];
  const h=harness('components/modals/AutoLabelModal.tsx',{
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
  const h=harness('components/modals/AutoLabelModal.tsx');
  let r=h.render(autoLabelProps);
  const input=classNameInput(r.tree,'01');
  assert.ok(input,'Class names must be editable');
  match(r.tree,n=>n.type==='button' && n.props.title==='Edit prompt text').props.onClick();
  r=h.render(autoLabelProps);
  find(r.tree,'textarea').props.onChange({target:{value:'a red octagonal stop sign beside the road'}});
  r=h.render(autoLabelProps);
  classNameInput(r.tree,'01').props.onChange({target:{value:'Stop sign'}});
  r=h.render(autoLabelProps);
  assert.equal(find(r.tree,'textarea').props.value,'a red octagonal stop sign beside the road');
});

test('new classes focus the name field and keep distinct identities after deletion',()=>{
  const h=harness('components/modals/AutoLabelModal.tsx');
  let r=h.render(autoLabelProps);
  match(r.tree,n=>n.type==='button' && content(n).includes('Add Class Prompt')).props.onClick();
  r=h.render(autoLabelProps);
  const added=classNameInput(r.tree,'02');
  assert.ok(added,'New classes need an editable name field');
  assert.equal(added.props.autoFocus,true);
  added.props.onChange({target:{value:'Helmet'}});
  r=h.render(autoLabelProps);
  match(r.tree,n=>n.type==='button' && n.props['aria-label']==='Remove class Car').props.onClick();
  r=h.render(autoLabelProps);
  match(r.tree,n=>n.type==='button' && content(n).includes('Add Class Prompt')).props.onClick();
  r=h.render(autoLabelProps);
  assert.equal(classNameInput(r.tree,'02').props.value,'Helmet');
  assert.ok(classNameInput(r.tree,'03'));
});

test('blank class names block labeling and surrounding whitespace is trimmed',async()=>{
  let previews=0,batches=0;
  const h=harness('components/modals/AutoLabelModal.tsx',{
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
