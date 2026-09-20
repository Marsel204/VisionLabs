import React from 'react';
import type { BoundingBox } from '../../types';
import { CLASS_COLORS } from '../../types';

interface Props {
  selectedBox: BoundingBox | null;
  onUpdateBox: (updated: BoundingBox) => void;
  onDeleteBox: (id: string) => void;
  onAcceptBox: (id: string) => void;
}

export const ObjectInspector: React.FC<Props> = ({
  selectedBox,
  onUpdateBox,
  onDeleteBox,
  onAcceptBox,
}) => {
  if (!selectedBox) {
    return (
      <aside className="w-[245px] flex-shrink-0 bg-[#0f1524]/85 backdrop-blur-2xl border-l border-[#2a3a48]/40 flex flex-col items-center justify-center p-4 text-center select-none z-20">
        <div className="w-10 h-10 rounded-xl bg-[#141c2e] border border-[#2a3a48]/40 flex items-center justify-center text-[#4a6070] mb-2">
          <span className="material-symbols-outlined text-[20px]">crop_square</span>
        </div>
        <p className="text-xs font-semibold text-[#a0b4c4]">No Object Selected</p>
        <p className="text-[10px] text-[#4a6070] mt-1">
          Click an existing bounding box or draw a new box on the canvas.
        </p>
      </aside>
    );
  }

  const colorConfig = CLASS_COLORS[selectedBox.class_name.toLowerCase()] || {
    stroke: '#06b6d4',
    bg: 'rgba(6, 182, 212, 0.15)',
    text: '#06b6d4',
  };

  return (
    <aside className="w-[245px] flex-shrink-0 bg-[#0f1524]/85 backdrop-blur-2xl border-l border-[#2a3a48]/40 flex flex-col z-20 shadow-2xl overflow-y-auto select-none">
      {/* Header */}
      <div className="p-2.5 bg-[#111828]/80 border-b border-[#2a3a48]/30 flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <span className="material-symbols-outlined text-[#06b6d4] text-[17px]">fact_check</span>
          <span className="font-semibold text-[11px] tracking-tight text-[#e0e8f0] uppercase">
            Object Inspector
          </span>
        </div>
        <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-[#10b981]/15 text-[#34d399] font-semibold">
          VALIDATED
        </span>
      </div>

      {/* Selected Box Info */}
      <div className="p-2.5 border-b border-[#2a3a48]/30 space-y-2.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span
              className="w-2.5 h-2.5 rounded-full shadow-sm"
              style={{ backgroundColor: colorConfig.stroke }}
            ></span>
            <span className="font-mono font-bold text-[11px] text-[#e0e8f0]">
              #{selectedBox.id.toUpperCase()}
            </span>
          </div>
          <span className="text-[9px] font-mono text-[#a0b4c4] bg-[#141c2e] px-1.5 py-0.5 rounded border border-[#2a3a48]/30">
            2D Box
          </span>
        </div>

        {/* Class Designation */}
        <div className="space-y-1">
          <label className="text-[9px] font-mono uppercase tracking-wider text-[#a0b4c4]">
            Class Designation
          </label>
          <select
            value={selectedBox.class_name.toLowerCase()}
            onChange={(e) =>
              onUpdateBox({ ...selectedBox, class_name: e.target.value })
            }
            className="w-full bg-[#0a0e1a] text-[#e0e8f0] text-[11px] px-2 py-1 rounded-md border border-[#2a3a48]/40 focus:outline-none focus:border-[#06b6d4]"
          >
            <option value="car">Car (Sedan / SUV)</option>
            <option value="motorcycle">Motorcycle (Scooter / Gojek)</option>
            <option value="minivan">Minivan (Angkot / Carry)</option>
            <option value="bus">Bus (Transit / City)</option>
            <option value="truck">Truck (Cargo / Flatbed)</option>
            <option value="person">Person (Pedestrian)</option>
          </select>
        </div>

        {/* Bounding Geometry */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[9px] font-mono text-[#a0b4c4]">
            <span>BOUNDING GEOMETRY (PX)</span>
          </div>
          <div className="grid grid-cols-4 gap-1 font-mono text-[10px]">
            <div className="bg-[#0a0e1a] p-1 rounded border border-[#2a3a48]/30 flex flex-col">
              <span className="text-[8px] text-[#a0b4c4]">X</span>
              <span className="text-[#06b6d4] font-semibold">{Math.round(selectedBox.x)}</span>
            </div>
            <div className="bg-[#0a0e1a] p-1 rounded border border-[#2a3a48]/30 flex flex-col">
              <span className="text-[8px] text-[#a0b4c4]">Y</span>
              <span className="text-[#06b6d4] font-semibold">{Math.round(selectedBox.y)}</span>
            </div>
            <div className="bg-[#0a0e1a] p-1 rounded border border-[#2a3a48]/30 flex flex-col">
              <span className="text-[8px] text-[#a0b4c4]">W</span>
              <span className="text-[#e0e8f0] font-semibold">{Math.round(selectedBox.width)}</span>
            </div>
            <div className="bg-[#0a0e1a] p-1 rounded border border-[#2a3a48]/30 flex flex-col">
              <span className="text-[8px] text-[#a0b4c4]">H</span>
              <span className="text-[#e0e8f0] font-semibold">{Math.round(selectedBox.height)}</span>
            </div>
          </div>
        </div>

        {/* Model Confidence */}
        <div className="p-2 rounded-md bg-[#0a0e1a] border border-[#2a3a48]/30 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <div className="w-5 h-5 rounded bg-[#0e4d6e] flex items-center justify-center text-[#7dd3fc]">
              <span className="material-symbols-outlined text-[13px]">psychology</span>
            </div>
            <div>
              <div className="text-[10px] font-semibold text-[#e0e8f0]">
                {Math.round(selectedBox.confidence * 100)}% Confidence
              </div>
              <div className="text-[8px] font-mono text-[#a0b4c4]">YOLO11n-VisionForge</div>
            </div>
          </div>
          <span className="text-[9px] font-mono text-[#10b981] font-bold">+2.1%</span>
        </div>
      </div>

      {/* Attributes */}
      <div className="p-2.5 border-b border-[#2a3a48]/30 space-y-2">
        <span className="font-semibold text-[10px] text-[#e0e8f0] uppercase tracking-wider">
          Perception Attributes
        </span>
        <div className="space-y-1.5 text-[10px]">
          {/* Occluded */}
          <div className="flex items-center justify-between p-1.5 rounded bg-[#0a0e1a] border border-[#2a3a48]/30">
            <span className="text-[#e0e8f0]">Occluded</span>
            <input
              type="checkbox"
              checked={selectedBox.occluded || false}
              onChange={(e) => onUpdateBox({ ...selectedBox, occluded: e.target.checked })}
              className="accent-[#06b6d4] cursor-pointer"
            />
          </div>

          {/* Truncated */}
          <div className="flex items-center justify-between p-1.5 rounded bg-[#0a0e1a] border border-[#2a3a48]/30">
            <span className="text-[#e0e8f0]">Truncated (Edge)</span>
            <input
              type="checkbox"
              checked={selectedBox.truncated || false}
              onChange={(e) => onUpdateBox({ ...selectedBox, truncated: e.target.checked })}
              className="accent-[#06b6d4] cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* Quality Consensus Action Triage */}
      <div className="p-2.5 space-y-2 flex-1">
        <span className="font-semibold text-[10px] text-[#e0e8f0] uppercase tracking-wider">
          Consensus Triage
        </span>
        <div className="grid grid-cols-3 gap-1">
          <button
            onClick={() => onAcceptBox(selectedBox.id)}
            type="button"
            className="py-1.5 px-1 rounded-md bg-[#10b981]/20 hover:bg-[#10b981]/30 text-[#34d399] text-[9.5px] font-semibold flex flex-col items-center justify-center gap-0.5 transition-all cursor-pointer border border-[#10b981]/30"
          >
            <span className="material-symbols-outlined text-[14px]">check_circle</span>
            <span>Accept [A]</span>
          </button>
          <button
            type="button"
            className="py-1.5 px-1 rounded-md bg-[#f59e0b]/20 hover:bg-[#f59e0b]/30 text-[#fbbf24] text-[9.5px] font-semibold flex flex-col items-center justify-center gap-0.5 transition-all cursor-pointer border border-[#f59e0b]/30"
          >
            <span className="material-symbols-outlined text-[14px]">flag</span>
            <span>Flag [R]</span>
          </button>
          <button
            onClick={() => onDeleteBox(selectedBox.id)}
            type="button"
            className="py-1.5 px-1 rounded-md bg-[#ef4444]/20 hover:bg-[#ef4444]/30 text-[#f87171] text-[9.5px] font-semibold flex flex-col items-center justify-center gap-0.5 transition-all cursor-pointer border border-[#ef4444]/30"
          >
            <span className="material-symbols-outlined text-[14px]">delete</span>
            <span>Reject [D]</span>
          </button>
        </div>
      </div>
    </aside>
  );
};
