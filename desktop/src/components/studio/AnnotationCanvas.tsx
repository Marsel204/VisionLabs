import React, { useRef, useState, useEffect } from 'react';
import type { AuditFinding, BoundingBox, ImageMeta, ToolType } from '../../types';
import { getClassColor, LOW_CONFIDENCE, LOW_CONFIDENCE_COLOR } from '../../types';
import { AUDIT_KIND_LABELS, ENHANCE_PRESETS } from '../../audit';
import { Icon, KeyHint } from '../ui';
import { getImageUrl } from '../../services/api';

interface Props {
  image: ImageMeta | null;
  boxes: BoundingBox[];
  selectedBoxId: string | null;
  activeTool: ToolType;
  imageIndex?: number;
  totalImages?: number;
  activeClassName?: string;
  availableClasses?: string[];
  readOnly?: boolean;
  onSelectBox: (id: string | null) => void;
  onAddBox: (box: BoundingBox) => void;
  onUpdateBox?: (box: BoundingBox) => void;
  onPrevImage?: () => void;
  onNextImage?: () => void;
  onRunYoloDetect: () => void;
  isDetecting: boolean;
  auditFindings?: AuditFinding[];
  selectedFindingId?: string | null;
  isAuditing?: boolean;
  onSelectFinding?: (id: string) => void;
  onRunAudit?: () => void;
  enhance?: string;
  onChangeEnhance?: (preset: string) => void;
  flipDetect?: boolean;
  flipDetectDisabled?: boolean;
  onChangeFlipDetect?: (enabled: boolean) => void;
  onSelectTool?: (tool: ToolType) => void;
  footer?: React.ReactNode;
}

const TOOLS: { type: ToolType; icon: string; key: string; title: string }[] = [
  { type: 'select', icon: 'near_me', key: 'V', title: 'Pilih & ubah' },
  { type: 'bbox', icon: 'crop_free', key: 'B', title: 'Kotak' },
  { type: 'polygon', icon: 'polyline', key: 'P', title: 'Poligon' },
  { type: 'pan', icon: 'pan_tool', key: 'H', title: 'Geser kanvas' },
];

const TOOL_HINTS: Record<ToolType, string> = {
  select: 'Pilih objek untuk mengedit',
  bbox: 'Tarik untuk menggambar kotak baru',
  polygon: 'Klik titik-titik, klik ganda untuk selesai',
  pan: 'Tarik untuk menggeser gambar',
};

function boxColor(box: BoundingBox) {
  return box.confidence < LOW_CONFIDENCE ? LOW_CONFIDENCE_COLOR : getClassColor(box.class_name);
}

export const AnnotationCanvas: React.FC<Props> = ({
  image,
  boxes,
  selectedBoxId,
  activeTool,
  activeClassName = 'object',
  availableClasses = [],
  readOnly = false,
  onUpdateBox,
  onSelectBox,
  onAddBox,
  onRunYoloDetect,
  isDetecting,
  auditFindings = [],
  selectedFindingId = null,
  isAuditing = false,
  onSelectFinding,
  onRunAudit,
  enhance = 'none',
  onChangeEnhance,
  flipDetect = false,
  flipDetectDisabled = false,
  onChangeFlipDetect,
  onSelectTool,
  footer,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });

  // Size the image and its overlay as one plane, using the space left by the toolbar/filmstrip.
  // Constraining only the wrapper's height lets the image overflow while the SVG gets squashed.
  useEffect(() => {
    const viewport = containerRef.current;
    if (!viewport) return;
    const measure = () => {
      setViewportSize({ width: viewport.clientWidth, height: viewport.clientHeight });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(viewport);
    return () => observer.disconnect();
  }, [image?.path, image?.image_id, image?.filename]);

  const [zoom, setZoom] = useState<number>(100);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawCurrent, setDrawCurrent] = useState<{ x: number; y: number } | null>(null);

  const [polygonPoints, setPolygonPoints] = useState<{ x: number; y: number }[]>([]);
  const transformRef = useRef<{ box: BoundingBox; start: { x: number; y: number }; corner?: number } | null>(null);
  const [transformPreview, setTransformPreview] = useState<BoundingBox | null>(null);
  const transformPreviewRef = useRef<BoundingBox | null>(null);
  const [labelOpacity, setLabelOpacity] = useState<number>(75);

  // Reset zoom & pan when image changes
  useEffect(() => {
    setZoom(100);
    setPan({ x: 0, y: 0 });
    setPolygonPoints([]); setIsDrawing(false); setDrawStart(null); setDrawCurrent(null);
    transformRef.current = null; setTransformPreview(null); transformPreviewRef.current = null;
  }, [image?.path, image?.image_id, image?.filename]);

  // Convert client coordinates to image pixel coordinates
  const clientToImageCoords = (clientX: number, clientY: number) => {
    if (!imgRef.current) return null;
    const rect = imgRef.current.getBoundingClientRect();
    const scaleX = (image?.width || 640) / rect.width;
    const scaleY = (image?.height || 640) / rect.height;

    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * scaleY;
    return {
      x: Math.max(0, Math.min(image?.width || 640, x)),
      y: Math.max(0, Math.min(image?.height || 640, y)),
    };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.button !== 0 && e.button !== 1) || readOnly) return;

    if (activeTool === 'pan' || e.button === 1) {
      setIsPanning(true);
      setStartPan({ x: e.clientX - pan.x, y: e.clientY - pan.y });
      return;
    }

    if (activeTool === 'polygon') {
      const point = clientToImageCoords(e.clientX, e.clientY);
      if (point) setPolygonPoints(previous => [...previous, point]);
      return;
    }
    if (activeTool === 'bbox') {
      const coords = clientToImageCoords(e.clientX, e.clientY);
      if (coords) {
        setIsDrawing(true);
        setDrawStart(coords);
        setDrawCurrent(coords);
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const transform = transformRef.current;
    if (transform && image) {
      const point = clientToImageCoords(e.clientX, e.clientY);
      if (!point) return;
      const b = transform.box;
      const dx = point.x - transform.start.x, dy = point.y - transform.start.y;
      let x1 = b.x, y1 = b.y, x2 = b.x + b.width, y2 = b.y + b.height;
      if (transform.corner === undefined) {
        x1 = Math.max(0, Math.min(image.width - b.width, b.x + dx));
        y1 = Math.max(0, Math.min(image.height - b.height, b.y + dy));
        x2 = x1 + b.width; y2 = y1 + b.height;
      } else {
        if (transform.corner === 0 || transform.corner === 2) x1 = Math.min(point.x, x2 - 1);
        else x2 = Math.max(point.x, x1 + 1);
        if (transform.corner < 2) y1 = Math.min(point.y, y2 - 1);
        else y2 = Math.max(point.y, y1 + 1);
      }
      const polygon = b.polygon_normalized?.map(([x, y]) => [
        (x1 + (x * image.width - b.x) * (x2 - x1) / b.width) / image.width,
        (y1 + (y * image.height - b.y) * (y2 - y1) / b.height) / image.height,
      ]);
      const updated = { ...b, x: x1, y: y1, width: x2-x1, height: y2-y1,
        norm_left: x1/image.width, norm_top: y1/image.height, norm_right: x2/image.width, norm_bottom: y2/image.height,
        polygon_normalized: polygon || b.polygon_normalized,
        polygon_pixels: polygon?.map(([x,y]) => [x*image.width, y*image.height]) || b.polygon_pixels };
      transformPreviewRef.current = updated; setTransformPreview(updated);
      return;
    }
    if (isPanning) {
      setPan({ x: e.clientX - startPan.x, y: e.clientY - startPan.y });
      return;
    }

    if (isDrawing && drawStart) {
      const coords = clientToImageCoords(e.clientX, e.clientY);
      if (coords) {
        setDrawCurrent(coords);
      }
    }
  };

  const handleMouseUp = () => {
    if (transformRef.current) {
      if (transformPreviewRef.current) onUpdateBox?.(transformPreviewRef.current);
      transformRef.current = null; transformPreviewRef.current = null; setTransformPreview(null);
      return;
    }
    if (isPanning) {
      setIsPanning(false);
    }

    if (isDrawing && drawStart && drawCurrent && image) {
      setIsDrawing(false);
      const x1 = Math.min(drawStart.x, drawCurrent.x);
      const y1 = Math.min(drawStart.y, drawCurrent.y);
      const x2 = Math.max(drawStart.x, drawCurrent.x);
      const y2 = Math.max(drawStart.y, drawCurrent.y);
      const w = x2 - x1;
      const h = y2 - y1;

      // Minimum size check
      if (w > 10 && h > 10) {
        const cls = activeClassName;
        const classId = availableClasses.indexOf(cls);
        if (classId < 0) { setDrawStart(null); setDrawCurrent(null); return; }
        const newBox: BoundingBox = {
          id: crypto.randomUUID(),
          class_name: cls,
          class_id: classId,
          confidence: 1.0,
          x: Math.round(x1),
          y: Math.round(y1),
          width: Math.round(w),
          height: Math.round(h),
          norm_left: Math.round((x1 / image.width) * 10000) / 10000,
          norm_top: Math.round((y1 / image.height) * 10000) / 10000,
          norm_right: Math.round((x2 / image.width) * 10000) / 10000,
          norm_bottom: Math.round((y2 / image.height) * 10000) / 10000,
          source: 'human',
        };
        onAddBox(newBox);
        onSelectBox(newBox.id);
      }
      setDrawStart(null);
      setDrawCurrent(null);
    }
  };

  const finishPolygon = () => {
    if (!image || readOnly) return;
    const points = polygonPoints.filter((point, i) => i === 0 || point.x !== polygonPoints[i-1].x || point.y !== polygonPoints[i-1].y);
    const classId = availableClasses.indexOf(activeClassName);
    if (points.length < 3 || classId < 0) return;
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const x = Math.min(...xs), y = Math.min(...ys), right = Math.max(...xs), bottom = Math.max(...ys);
    if (right <= x || bottom <= y) return;
    const polygon = points.map(p => [p.x/image.width, p.y/image.height]);
    const box: BoundingBox = { id: crypto.randomUUID(), class_name: activeClassName, class_id: classId,
      confidence: 1, source: 'human', x, y, width: right-x, height: bottom-y,
      norm_left: x/image.width, norm_top: y/image.height, norm_right: right/image.width, norm_bottom: bottom/image.height,
      polygon_normalized: polygon, polygon_pixels: points.map(p => [p.x,p.y]) };
    onAddBox(box); onSelectBox(box.id); setPolygonPoints([]);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -10 : 10;
    setZoom((prev) => Math.max(50, Math.min(300, prev + delta)));
  };

  if (!image) {
    return (
      <main className="flex flex-1 items-center justify-center rounded-xl border border-line bg-white text-sm text-ink-500">
        Memuat kanvas…
      </main>
    );
  }

  // Labels and handles are drawn in image pixels, so scale them up for large images.
  const ui = Math.max(1, image.width / 900);
  const fitScale = Math.min(1, viewportSize.width * .96 / image.width, viewportSize.height * .94 / image.height);
  const toolButton = (active: boolean) =>
    `flex h-8 items-center gap-1.5 rounded-md px-2 text-[11px] font-medium transition-colors cursor-pointer ${
      active ? 'bg-brand-100 text-brand-600 ring-1 ring-brand-200' : 'text-ink-500 hover:bg-brand-50 hover:text-brand-600'
    }`;

  return (
    <main
      className="flex min-h-[520px] min-w-0 flex-1 flex-col overflow-hidden rounded-xl border border-line bg-white select-none xl:min-h-0"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      {/* Toolbar */}
      <div className="flex min-h-12 shrink-0 flex-wrap items-center gap-1 border-b border-line px-3 py-1">
        {TOOLS.map(tool => (
          <button
            key={tool.type}
            type="button"
            title={`${tool.title} (${tool.key})`}
            aria-pressed={activeTool === tool.type}
            onClick={() => onSelectTool?.(tool.type)}
            className={toolButton(activeTool === tool.type)}
          >
            <Icon name={tool.icon} className="text-[17px]" />
            <span>{tool.key}</span>
          </button>
        ))}
        <span className="mx-1.5 h-6 w-px bg-line" />
        <span className="hidden min-w-0 truncate text-xs text-ink-500 2xl:inline">{readOnly ? 'Memuat anotasi…' : TOOL_HINTS[activeTool]}</span>

        <div className="ml-auto flex shrink-0 flex-wrap items-center gap-2 whitespace-nowrap">
          {onChangeEnhance && (
            <label
              title="Filter tampilan untuk frame yang sulit dilihat; label dan deteksi tidak berubah (F)"
              className="flex items-center gap-1 text-[11px] text-ink-500"
            >
              <Icon name="deblur" className="text-[16px]" />
              <select
                value={enhance}
                onChange={(e) => { onChangeEnhance(e.target.value); e.currentTarget.blur(); }}
                className="h-8 rounded-md border border-line-strong bg-white px-1.5 text-[11px] text-ink-700 cursor-pointer focus:outline-none focus:border-brand-500"
              >
                {ENHANCE_PRESETS.map((preset) => (
                  <option key={preset.id} value={preset.id}>{preset.label}</option>
                ))}
              </select>
            </label>
          )}
          {onChangeFlipDetect && (
            <label
              title={flipDetectDisabled
                ? 'Lapisan tambahan hanya berjalan di mesin lokal'
                : 'Deteksi juga pada salinan cermin lalu gabungkan: menemukan lebih banyak objek tertutup, sekitar 2x lebih lambat'}
              className={`flex items-center gap-1 text-[11px] text-ink-500 ${flipDetectDisabled ? 'opacity-50' : 'cursor-pointer'}`}
            >
              <input
                type="checkbox"
                checked={flipDetect}
                disabled={flipDetectDisabled}
                onChange={(e) => onChangeFlipDetect(e.target.checked)}
                className="accent-brand-600"
              />
              +Flip
            </label>
          )}
          <button
            onClick={onRunYoloDetect}
            disabled={isDetecting || readOnly}
            type="button"
            title="Jalankan deteksi YOLO pada gambar ini (R)"
            className="flex h-8 items-center gap-1.5 rounded-md border border-brand-200 bg-brand-50 px-2.5 text-[11px] font-semibold text-brand-600 hover:bg-brand-100 cursor-pointer disabled:opacity-50"
          >
            <Icon name={isDetecting ? 'progress_activity' : 'auto_awesome'} className={`text-[16px] ${isDetecting ? 'animate-spin' : ''}`} />
            {isDetecting ? 'Mendeteksi…' : 'Deteksi AI'}
          </button>
          {onRunAudit && (
            <button
              onClick={onRunAudit}
              disabled={isAuditing}
              type="button"
              title="Cari objek terlewat, tertutup, dan kotak yang salah (Q)"
              className="flex h-8 items-center gap-1.5 rounded-md border border-line-strong bg-white px-2.5 text-[11px] font-semibold text-ink-700 hover:bg-brand-50 cursor-pointer disabled:opacity-50"
            >
              <Icon name={isAuditing ? 'progress_activity' : 'rule'} className={`text-[16px] ${isAuditing ? 'animate-spin' : ''}`} />
              {isAuditing ? 'Memeriksa…' : 'Audit'}
            </button>
          )}
          <span className="hidden pl-1 text-[11px] tabular-nums text-ink-400 2xl:inline">{image.width} × {image.height}</span>
        </div>
      </div>

      {/* Viewport */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onWheel={handleWheel}
        onDoubleClick={() => { if (activeTool === "polygon") finishPolygon(); }}
        className={`relative flex min-h-[320px] flex-1 items-center justify-center overflow-hidden bg-canvas ${
          activeTool === 'pan' ? 'cursor-grab active:cursor-grabbing' : 'cursor-crosshair'
        }`}
      >
        <div
          style={{
            width: image.width * fitScale,
            height: image.height * fitScale,
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom / 100})`,
            transformOrigin: 'center center',
          }}
          className="relative shrink-0 select-none transition-transform duration-75 ease-out"
        >
          <img
            ref={imgRef}
            src={getImageUrl(image.path || image.image_id || image.filename, enhance)}
            alt={image.filename}
            className="pointer-events-none block h-full w-full"
            draggable={false}
          />

          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 ${image.width} ${image.height}`}
            preserveAspectRatio="none"
          >
            {boxes.map((originalBox) => {
              const box = transformPreview?.id === originalBox.id ? transformPreview : originalBox;
              const isSelected = box.id === selectedBoxId;
              const color = boxColor(box);
              const label = `${box.class_name} ${box.confidence.toFixed(2)}`;
              const font = 11 * ui;

              return (
                <g
                  key={box.id}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectBox(box.id);
                  }}
                  className="pointer-events-auto cursor-pointer"
                >
                  {box.polygon_normalized && <polygon
                    points={box.polygon_normalized.map(([x,y]) => `${x*image.width},${y*image.height}`).join(' ')}
                    fill={color.bg} stroke={color.stroke} strokeWidth={1.5 * ui} />}
                  <rect
                    x={box.x}
                    y={box.y}
                    width={box.width}
                    height={box.height}
                    fill={color.stroke}
                    fillOpacity={(labelOpacity / 100) * (isSelected ? 0.2 : 0.08)}
                    stroke={color.stroke}
                    strokeWidth={(isSelected ? 2.5 : 1.5) * ui}
                    onMouseDown={(e) => {
                      if (activeTool !== 'select' || readOnly) return;
                      e.stopPropagation(); const point = clientToImageCoords(e.clientX, e.clientY);
                      if (point) { onSelectBox(box.id); transformRef.current = { box, start: point }; }
                    }}
                  />

                  {isSelected && activeTool === 'select' && !readOnly && [
                    [box.x, box.y], [box.x + box.width, box.y], [box.x, box.y + box.height], [box.x + box.width, box.y + box.height]
                  ].map(([x,y], corner) => <rect key={corner} x={x-3*ui} y={y-3*ui} width={6*ui} height={6*ui}
                    fill="#ffffff" stroke={color.stroke} strokeWidth={1.5 * ui}
                    onMouseDown={e => { e.stopPropagation(); const point = clientToImageCoords(e.clientX, e.clientY);
                      if (point) transformRef.current = { box, start: point, corner }; }} />)}

                  <rect
                    x={box.x - 0.75 * ui}
                    y={Math.max(0, box.y - font * 1.6)}
                    width={label.length * font * 0.6 + font}
                    height={font * 1.6}
                    fill={color.stroke}
                    fillOpacity={Math.max(0.55, labelOpacity / 100)}
                  />
                  <text
                    x={box.x + font * 0.45}
                    y={Math.max(font * 1.15, box.y - font * 0.45)}
                    fill="#ffffff"
                    fontFamily="Manrope, sans-serif"
                    fontSize={font}
                    fontWeight={600}
                  >
                    {label}
                  </text>
                </g>
              );
            })}

            {/* Audit findings: dashed ghosts for suggestions, dashed outlines on suspect boxes */}
            {auditFindings.map((finding) => {
              const meta = AUDIT_KIND_LABELS[finding.kind];
              const selected = finding.id === selectedFindingId;
              const subject = boxes.find((b) => b.id === finding.annotation_id);
              const rect = finding.box && finding.kind !== 'class_mismatch'
                ? finding.box
                : subject ?? null;
              const outline = finding.kind === 'misaligned' && subject ? subject : null;
              if (!rect) return null;
              return (
                <g key={finding.id} className="pointer-events-none">
                  {outline && (
                    <rect x={outline.x} y={outline.y} width={outline.width} height={outline.height}
                      fill="none" stroke={meta.color} strokeWidth={1.5 * ui} strokeDasharray={`${2 * ui} ${3 * ui}`} />
                  )}
                  <rect
                    x={rect.x} y={rect.y} width={rect.width} height={rect.height}
                    fill={finding.kind === 'possibly_missed' ? `${meta.color}33` : 'none'}
                    stroke={meta.color} strokeWidth={(selected ? 3 : 2) * ui} strokeDasharray={`${8 * ui} ${4 * ui}`}
                    className="pointer-events-auto cursor-pointer"
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={(e) => { e.stopPropagation(); onSelectFinding?.(finding.id); }}
                  />
                  <text x={rect.x + 4 * ui} y={rect.y + 13 * ui} fill={meta.color} fontFamily="Manrope, sans-serif"
                    fontSize={11 * ui} fontWeight="bold" stroke="#14243d" strokeWidth={3 * ui} paintOrder="stroke">
                    {meta.label}{finding.class_name ? ` · ${finding.class_name}` : ''}
                  </text>
                </g>
              );
            })}

            {polygonPoints.length > 0 && <polyline points={polygonPoints.map(p => `${p.x},${p.y}`).join(' ')}
              fill="rgba(47,111,228,0.2)" stroke="#2f6fe4" strokeWidth={2 * ui} />}
            {isDrawing && drawStart && drawCurrent && (
              <rect
                x={Math.min(drawStart.x, drawCurrent.x)}
                y={Math.min(drawStart.y, drawCurrent.y)}
                width={Math.abs(drawCurrent.x - drawStart.x)}
                height={Math.abs(drawCurrent.y - drawStart.y)}
                fill="rgba(47, 111, 228, 0.18)"
                stroke="#4a84ea"
                strokeWidth={2 * ui}
                strokeDasharray={`${4 * ui} ${2 * ui}`}
              />
            )}
          </svg>
        </div>

        {activeTool === 'polygon' && polygonPoints.length > 0 && (
          <div className="absolute left-3 top-3 z-10 flex items-center gap-3 rounded-md bg-white px-3 py-1.5 text-xs text-ink-700 shadow">
            {polygonPoints.length} titik
            <button type="button" onMouseDown={e => e.stopPropagation()} onClick={finishPolygon}
              className="font-semibold text-brand-600 cursor-pointer">Selesai</button>
            <button type="button" onMouseDown={e => e.stopPropagation()} onClick={() => setPolygonPoints([])}
              className="text-ink-500 cursor-pointer">Batal</button>
          </div>
        )}
      </div>

      {/* Hint row with view controls */}
      <div className="flex h-11 shrink-0 items-center gap-4 border-t border-line px-3 text-[11px] text-ink-500">
        <span className="flex min-w-0 items-center gap-2 truncate">
          <Icon name="center_focus_weak" className="text-[16px]" />
          Geser kotak untuk memperbaiki posisi. Ubah kelas dari panel anotasi.
        </span>
        <div className="ml-auto flex shrink-0 items-center gap-3">
          <span className="flex items-center gap-1">
            <button type="button" aria-label="Perkecil" onClick={() => setZoom(z => Math.max(50, z - 10))}
              className="flex h-6 w-6 items-center justify-center rounded hover:bg-brand-50 cursor-pointer"><Icon name="remove" className="text-[15px]" /></button>
            <button type="button" title="Sesuaikan layar" onClick={() => { setZoom(100); setPan({ x: 0, y: 0 }); }}
              className="w-10 text-center tabular-nums hover:text-brand-600 cursor-pointer">{zoom}%</button>
            <button type="button" aria-label="Perbesar" onClick={() => setZoom(z => Math.min(300, z + 10))}
              className="flex h-6 w-6 items-center justify-center rounded hover:bg-brand-50 cursor-pointer"><Icon name="add" className="text-[15px]" /></button>
          </span>
          <label className="flex items-center gap-1.5" title="Kepekatan label">
            <Icon name="opacity" className="text-[15px]" />
            <input type="range" min="10" max="100" value={labelOpacity}
              onChange={(e) => setLabelOpacity(Number(e.target.value))}
              onMouseUp={(e) => e.currentTarget.blur()}
              className="w-16 accent-brand-600 cursor-pointer" />
          </label>
          <KeyHint k="B" label="Kotak" />
          <KeyHint k="V" label="Pilih" />
        </div>
      </div>

      {footer}
    </main>
  );
};
