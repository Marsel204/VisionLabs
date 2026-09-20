import React, { useRef, useState, useEffect } from 'react';
import type { BoundingBox, ImageMeta, ToolType } from '../../types';
import { getClassColor } from '../../types';
import { getImageUrl } from '../../services/api';

interface Props {
  image: ImageMeta | null;
  boxes: BoundingBox[];
  selectedBoxId: string | null;
  activeTool: ToolType;
  imageIndex: number;
  totalImages: number;
  activeClassName?: string;
  onSelectBox: (id: string | null) => void;
  onAddBox: (box: BoundingBox) => void;
  onUpdateBox?: (box: BoundingBox) => void;
  onPrevImage: () => void;
  onNextImage: () => void;
  onRunYoloDetect: () => void;
  isDetecting: boolean;
}

export const AnnotationCanvas: React.FC<Props> = ({
  image,
  boxes,
  selectedBoxId,
  activeTool,
  imageIndex,
  totalImages,
  activeClassName = 'object',
  onSelectBox,
  onAddBox,
  onPrevImage,
  onNextImage,
  onRunYoloDetect,
  isDetecting,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const [zoom, setZoom] = useState<number>(100);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawCurrent, setDrawCurrent] = useState<{ x: number; y: number } | null>(null);

  const [labelOpacity, setLabelOpacity] = useState<number>(75);

  // Reset zoom & pan when image changes
  useEffect(() => {
    setZoom(100);
    setPan({ x: 0, y: 0 });
  }, [image?.filename]);

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
    if (e.button !== 0) return; // Only left click

    if (activeTool === 'pan' || e.buttons === 4) {
      setIsPanning(true);
      setStartPan({ x: e.clientX - pan.x, y: e.clientY - pan.y });
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
        const cls = (activeClassName || 'object').toLowerCase();
        const classIds: Record<string, number> = {
          object: 0,
          motorcycle: 0,
          car: 1,
          bus: 2,
          truck: 3,
          minivan: 4,
          person: 5,
        };
        const newBox: BoundingBox = {
          id: `box-${Date.now().toString().slice(-4)}`,
          class_name: cls,
          class_id: classIds[cls] ?? 0,
          confidence: 0.95,
          x: Math.round(x1),
          y: Math.round(y1),
          width: Math.round(w),
          height: Math.round(h),
          norm_left: Math.round((x1 / image.width) * 10000) / 10000,
          norm_top: Math.round((y1 / image.height) * 10000) / 10000,
          norm_right: Math.round((x2 / image.width) * 10000) / 10000,
          norm_bottom: Math.round((y2 / image.height) * 10000) / 10000,
          source: 'manual',
        };
        onAddBox(newBox);
        onSelectBox(newBox.id);
      }
      setDrawStart(null);
      setDrawCurrent(null);
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -10 : 10;
    setZoom((prev) => Math.max(50, Math.min(300, prev + delta)));
  };

  if (!image) {
    return (
      <main className="flex-1 flex items-center justify-center bg-[#0a0e1a] text-[#a0b4c4]">
        Loading image canvas...
      </main>
    );
  }

  return (
    <main
      className="flex-1 flex flex-col relative overflow-hidden bg-[#0a0e1a] select-none"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      {/* Top Status & Controls Header */}
      <div className="h-8 px-3.5 bg-[#0f1524]/90 backdrop-blur-md flex items-center justify-between z-20 border-b border-[#2a3a48]/30">
        <div className="flex items-center gap-2 text-[10px] font-mono">
          <span className="px-1.5 py-0.5 rounded bg-[#141c2e] text-[#06b6d4] font-semibold border border-[#2a3a48]/30">
            {image.filename}
          </span>
          <span className="text-[#4a6070]">•</span>
          <span className="text-[#a0b4c4]">
            {image.width}×{image.height} (Full HD Static JPG)
          </span>
          <span className="text-[#4a6070]">•</span>
          <span className="text-[#10b981] flex items-center gap-1 font-semibold">
            <span className="w-1.5 h-1.5 rounded-full bg-[#10b981]"></span>
            {boxes.length} annotations
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* YOLO AI Detect Button */}
          <button
            onClick={onRunYoloDetect}
            disabled={isDetecting}
            type="button"
            className="flex items-center gap-1 px-2.5 py-0.5 rounded bg-[#06b6d4]/20 hover:bg-[#06b6d4]/30 text-[#06b6d4] text-[10px] font-mono font-semibold border border-[#06b6d4]/40 transition-all cursor-pointer disabled:opacity-50"
          >
            <span className="material-symbols-outlined text-[13px]">
              {isDetecting ? 'sync' : 'auto_awesome'}
            </span>
            <span>{isDetecting ? 'Detecting...' : 'Detect YOLO11'}</span>
          </button>
        </div>
      </div>

      {/* Main Canvas Viewport Area */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onWheel={handleWheel}
        className={`relative flex-1 w-full h-full overflow-hidden flex items-center justify-center ${
          activeTool === 'pan' ? 'cursor-grab active:cursor-grabbing' : 'cursor-crosshair'
        }`}
      >
        {/* Subtle background grid pattern */}
        <div className="absolute inset-0 bg-[radial-gradient(#1a2438_1px,transparent_1px)] [background-size:24px_24px] opacity-60 pointer-events-none"></div>

        {/* Viewport Transform Frame */}
        <div
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom / 100})`,
            transformOrigin: 'center center',
          }}
          className="relative max-w-[94%] max-h-[86%] shadow-2xl rounded-lg overflow-hidden transition-transform duration-75 ease-out select-none border border-[#2a3a48]/40"
        >
          {/* Active Image Viewport */}
          <img
            ref={imgRef}
            src={getImageUrl(image.filename)}
            alt={image.filename}
            className="w-auto h-auto max-w-full max-h-[72vh] object-contain pointer-events-none block"
            draggable={false}
          />

          {/* SVG 2D Vector Bounding Box Overlay Layer */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 ${image.width} ${image.height}`}
            preserveAspectRatio="none"
          >
            {boxes.map((box) => {
              const isSelected = box.id === selectedBoxId;
              const color = getClassColor(box.class_name);

              return (
                <g
                  key={box.id}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectBox(box.id);
                  }}
                  className="pointer-events-auto cursor-pointer group"
                >
                  {/* Bounding Box Rectangle */}
                  <rect
                    x={box.x}
                    y={box.y}
                    width={box.width}
                    height={box.height}
                    fill={color.bg}
                    fillOpacity={labelOpacity / 100}
                    stroke={color.stroke}
                    strokeWidth={isSelected ? 2.5 : 1.5}
                    rx={3}
                  />

                  {/* Corner Handles for selected box */}
                  {isSelected && (
                    <>
                      <rect x={box.x - 3} y={box.y - 3} width={6} height={6} fill={color.stroke} />
                      <rect x={box.x + box.width - 3} y={box.y - 3} width={6} height={6} fill={color.stroke} />
                      <rect x={box.x - 3} y={box.y + box.height - 3} width={6} height={6} fill={color.stroke} />
                      <rect x={box.x + box.width - 3} y={box.y + box.height - 3} width={6} height={6} fill={color.stroke} />
                    </>
                  )}

                  {/* Label Header Pill */}
                  <rect
                    x={box.x}
                    y={Math.max(0, box.y - 18)}
                    width={Math.max(70, box.class_name.length * 8 + 36)}
                    height={18}
                    fill="#0f1524"
                    fillOpacity={0.92}
                    stroke={color.stroke}
                    strokeWidth={1}
                    rx={3}
                  />
                  <text
                    x={box.x + 5}
                    y={Math.max(12, box.y - 5)}
                    fill={color.text}
                    fontFamily="monospace"
                    fontSize="11"
                    fontWeight="bold"
                  >
                    {box.class_name} {Math.round(box.confidence * 100)}%
                  </text>
                </g>
              );
            })}

            {/* Drawing Preview Rectangle */}
            {isDrawing && drawStart && drawCurrent && (
              <rect
                x={Math.min(drawStart.x, drawCurrent.x)}
                y={Math.min(drawStart.y, drawCurrent.y)}
                width={Math.abs(drawCurrent.x - drawStart.x)}
                height={Math.abs(drawCurrent.y - drawStart.y)}
                fill="rgba(6, 182, 212, 0.2)"
                stroke="#06b6d4"
                strokeWidth={2}
                strokeDasharray="4 2"
              />
            )}
          </svg>
        </div>
      </div>

      {/* Floating Bottom Static Image Navigation Bar */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#0f1524]/90 backdrop-blur-2xl border border-[#2a3a48]/40 shadow-2xl z-30 select-none">
        {/* Pagination Prev/Next */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={onPrevImage}
            disabled={imageIndex === 0}
            type="button"
            className="flex items-center gap-1 px-2.5 py-1 rounded-full bg-[#141c2e] hover:bg-[#1a2438] text-[#e0e8f0] text-[11px] font-medium transition-colors disabled:opacity-30 cursor-pointer"
            title="Previous Image (A / [)"
          >
            <span className="material-symbols-outlined text-[14px]">arrow_back</span>
            <span>Prev</span>
          </button>
          <span className="font-mono text-[11px] font-medium text-[#e0e8f0] px-2 whitespace-nowrap">
            Image <strong className="text-[#06b6d4] font-semibold">{imageIndex + 1}</strong> of{' '}
            {totalImages}
          </span>
          <button
            onClick={onNextImage}
            disabled={imageIndex >= totalImages - 1}
            type="button"
            className="flex items-center gap-1 px-2.5 py-1 rounded-full bg-[#141c2e] hover:bg-[#1a2438] text-[#e0e8f0] text-[11px] font-medium transition-colors disabled:opacity-30 cursor-pointer"
            title="Next Image (D / ])"
          >
            <span>Next</span>
            <span className="material-symbols-outlined text-[14px]">arrow_forward</span>
          </button>
        </div>

        <span className="w-px h-4 bg-[#2a3a48]"></span>

        {/* Quick Zoom Presets */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setZoom(100)}
            type="button"
            className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors cursor-pointer ${
              zoom === 100
                ? 'bg-[#0e4d6e] text-[#7dd3fc] font-semibold'
                : 'text-[#a0b4c4] hover:text-[#e0e8f0] hover:bg-[#141c2e]'
            }`}
          >
            100%
          </button>
          <button
            onClick={() => setZoom(140)}
            type="button"
            className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors cursor-pointer ${
              zoom === 140
                ? 'bg-[#0e4d6e] text-[#7dd3fc] font-semibold'
                : 'text-[#a0b4c4] hover:text-[#e0e8f0] hover:bg-[#141c2e]'
            }`}
          >
            140%
          </button>
          <button
            onClick={() => {
              setZoom(100);
              setPan({ x: 0, y: 0 });
            }}
            type="button"
            className="w-6 h-6 rounded bg-[#141c2e] hover:bg-[#1a2438] text-[#a0b4c4] hover:text-[#7dd3fc] flex items-center justify-center transition-colors cursor-pointer"
            title="Fit to Screen"
          >
            <span className="material-symbols-outlined text-[14px]">aspect_ratio</span>
          </button>
        </div>

        <span className="w-px h-4 bg-[#2a3a48]"></span>

        {/* Label Opacity Slider */}
        <div className="flex items-center gap-1.5 text-xs">
          <span className="material-symbols-outlined text-[14px] text-[#a0b4c4]" title="Label Opacity">
            opacity
          </span>
          <input
            type="range"
            min="10"
            max="100"
            value={labelOpacity}
            onChange={(e) => setLabelOpacity(Number(e.target.value))}
            className="w-14 h-1.5 bg-[#1a2438] rounded-lg appearance-none cursor-pointer accent-[#06b6d4]"
          />
          <span className="font-mono text-[10px] text-[#06b6d4] w-6">{labelOpacity}%</span>
        </div>
      </div>
    </main>
  );
};
