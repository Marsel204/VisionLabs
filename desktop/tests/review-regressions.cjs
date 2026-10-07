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
const jsx = (type, props) => ({type, props:props || {}});
function harness(file, supplied = {}) {
  const state=[], refs=[], effects=[], callbacks=[]; let cursor=0, pending=[];
  const same=(a,b)=>a && b && a.length===b.length && a.every((v,i)=>Object.is(v,b[i]));
  const React={
    useState(initial) { const i=cursor++; if (!(i in state)) state[i]=typeof initial==='function'?initial():initial;
      return [state[i], v=>{state[i]=typeof v==='function'?v(state[i]):v;}]; },
    useRef(initial) { const i=cursor++; return refs[i] || (refs[i]={current:initial}); },
    useCallback(fn,deps) { const i=cursor++; if (!callbacks[i] || !same(deps,callbacks[i].deps)) callbacks[i]={fn,deps}; return callbacks[i].fn; },
    useEffect(fn,deps) { const i=cursor++; if (!effects[i] || !same(deps,effects[i].deps)) {
      const old=effects[i]; effects[i]={deps,cleanup:null}; pending.push(()=>{old?.cleanup?.(); effects[i].cleanup=fn();}); } },
  };
  const api={imageIdentity:img=>img.path||img.image_id||img.filename,
    fetchHealth:async()=>({dataset_dir:'/dataset',classes:['dog','cat']}),
    fetchImages:async()=>({images:[],total:0,directory:'/dataset'}),
    fetchAllImages:async()=>({images:[],total:0,directory:'/dataset'}),
    fetchAnnotations:async()=>({boxes:[],revision:0}),
    saveAnnotations:async(_n,boxes,revision)=>({boxes,revision:(revision||0)+1}),
    getImageUrl:n=>n,...supplied};
  const code=ts.transpileModule(fs.readFileSync(path.join(root,file),'utf8'), {
    compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,target:ts.ScriptTarget.ES2022},
  }).outputText;
  const exports={}; let nextTimer=0;
  vm.runInNewContext(code, {exports, require(name) {
    if (name==='react') return {...React,default:React};
    if (name==='react/jsx-runtime') return {jsx,jsxs:jsx};
    if (name.endsWith('/api')) return api;
    if (name.endsWith('/types')) return {getClassColor:()=>({stroke:'blue',text:'blue',bg:'black'})};
    return new Proxy({}, {get:(_t,key)=>String(key)});
  }, window:{location:{search:''},addEventListener(){},removeEventListener(){}}, URLSearchParams,console,
  crypto:webcrypto,setTimeout:()=>++nextTimer,clearTimeout(){} });
  const component=exports.default || exports[Object.keys(exports)[0]];
  return {state,refs,render(props) {cursor=0;pending=[];const tree=component(props);
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
