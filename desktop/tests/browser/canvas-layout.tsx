import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { AnnotationCanvas } from '../../src/components/studio/AnnotationCanvas';
import { ImageWithBoxes } from '../../src/components/ImageWithBoxes';
import type { BoundingBox, ImageMeta, ToolType } from '../../src/types';
import '../../src/index.css';

const params = new URLSearchParams(location.search);
const width = Number(params.get('width') || 640);
const height = Number(params.get('height') || 640);
const image: ImageMeta = {
  filename: `${width}x${height}.svg`, width, height, size_bytes: 0,
  annotation_count: 1, status: 'unreviewed',
};
const initial: BoundingBox = {
  id: 'bus', class_name: 'bus', class_id: 0, confidence: 1,
  x: width * .29, y: height * .27, width: width * .675, height: height * .55,
  norm_left: .29, norm_top: .27, norm_right: .965, norm_bottom: .82,
  polygon_normalized: [[.29, .27], [.965, .27], [.965, .82], [.29, .82]],
};

function Fixture() {
  const [boxes, setBoxes] = useState([initial]);
  const [selected, setSelected] = useState<string | null>('bus');
  const [tool, setTool] = useState<ToolType>('select');
  const [enhance, setEnhance] = useState('none');
  return <>
    <div id="canvas-fixture" className="flex" style={{ width: 'calc(100vw - 320px)', height: 'calc(100vh - 220px)', margin: '100px 20px 0' }}>
      <AnnotationCanvas image={image} boxes={boxes} selectedBoxId={selected} activeTool={tool}
        activeClassName="bus" availableClasses={['bus']} onSelectBox={setSelected}
        onAddBox={box => setBoxes(previous => [...previous, box])}
        onUpdateBox={box => setBoxes(previous => previous.map(item => item.id === box.id ? box : item))}
        onSelectTool={setTool} onRunYoloDetect={() => {}} isDetecting={false}
        enhance={enhance} onChangeEnhance={setEnhance}
        footer={<div className="h-[60px] shrink-0">Filmstrip</div>} />
    </div>
    <output id="boxes" hidden>{JSON.stringify(boxes)}</output>
    <div id="preview-fixture" style={{ width: 700 }}>
      <ImageWithBoxes src={`/api/image/${image.filename}`} alt="Bus preview" width={width} height={height}
        boxes={[initial]} className="max-h-[360px] overflow-hidden" imgClassName="max-h-[360px] object-cover" />
    </div>
  </>;
}

createRoot(document.getElementById('root')!).render(<Fixture />);
