import React from 'react';
import type { SystemHealth } from '../../types';

interface Props {
  health: SystemHealth | null;
  imageCount: number;
  onOpenAutoLabel: () => void;
  onOpenExport: () => void;
  onOpenImport: () => void;
  onOpenShortcuts?: () => void;
}

export const StudioHeader: React.FC<Props> = ({
  health,
  imageCount,
  onOpenAutoLabel,
  onOpenExport,
  onOpenImport,
  onOpenShortcuts,
}) => {
  return (
    <header className="h-12 bg-[#0f1524]/90 backdrop-blur-xl border-b border-[#2a3a48]/40 flex items-center justify-between px-3.5 z-40 select-none shadow-sm">
      {/* Left: Brand & Dataset Dropdown */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 pr-3 border-r border-[#2a3a48]/40">
          <div className="w-7 h-7 rounded-lg bg-[#0e4d6e] border border-[#7dd3fc]/30 flex items-center justify-center text-[#7dd3fc] shadow-sm">
            <span className="material-symbols-outlined text-[17px]">view_in_ar</span>
          </div>
          <span className="font-semibold text-xs tracking-tight text-[#e0e8f0]">VisionStudio</span>
          <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-[#06b6d4]/20 text-[#06b6d4]">AI</span>
        </div>

        {/* Dataset Badges */}
        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-[#111828] text-[11px] text-[#a0b4c4] font-mono border border-[#2a3a48]/20">
          <span className="w-1.5 h-1.5 rounded-full bg-[#10b981]"></span>
          <span className="text-[#e0e8f0]">{imageCount} Images (JPG)</span>
        </div>

        <div className="hidden xl:flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-[#111828] text-[11px] text-[#a0b4c4] font-mono border border-[#2a3a48]/20">
          <span className="w-1.5 h-1.5 rounded-full bg-[#06b6d4] animate-pulse"></span>
          <span>Model: YOLO11n-VisionForge</span>
        </div>
      </div>

      {/* Right: Telemetry & Actions */}
      <div className="flex items-center gap-2.5">
        {/* GPU Status */}
        <div className="hidden md:flex items-center gap-2 px-2.5 py-1 rounded-full bg-[#111828] text-[11px] font-mono text-[#a0b4c4] border border-[#2a3a48]/30">
          <span className="material-symbols-outlined text-[14px] text-[#7dd3fc]">memory</span>
          <span className="text-[#e0e8f0]">{health?.gpu?.device || 'RTX 4090'}</span>
          <span className="text-[#4a6070]">•</span>
          <span className="text-[#10b981] font-semibold">{health?.gpu?.temperature || '41°C'}</span>
          <span className="text-[#4a6070]">•</span>
          <span className="text-[#7dd3fc]">Ready</span>
        </div>

        {/* Auto-Label AI Trigger */}
        <button
          onClick={onOpenAutoLabel}
          type="button"
          className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-gradient-to-r from-[#0e4d6e] to-[#3d2060] text-[#e0e8f0] text-xs font-semibold hover:brightness-110 shadow-sm border border-[#7dd3fc]/30 transition-all cursor-pointer"
        >
          <span className="material-symbols-outlined text-[15px] text-[#7dd3fc]">auto_awesome</span>
          <span>Auto-Label AI</span>
        </button>

        {/* Import Trigger */}
        <button
          onClick={onOpenImport}
          type="button"
          className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-[#141c2e] hover:bg-[#1a2438] text-xs font-medium text-[#e0e8f0] border border-[#2a3a48]/40 hover:border-[#10b981]/50 transition-colors cursor-pointer"
          title="Import images or switch dataset directory"
        >
          <span className="material-symbols-outlined text-[15px] text-[#10b981]">file_upload</span>
          <span>Import</span>
        </button>

        {/* Export Trigger */}
        <button
          onClick={onOpenExport}
          type="button"
          className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-[#141c2e] hover:bg-[#1a2438] text-xs font-medium text-[#e0e8f0] border border-[#2a3a48]/40 hover:border-[#06b6d4]/50 transition-colors cursor-pointer"
          title="Export dataset to YOLO or COCO formats (E)"
        >
          <span className="material-symbols-outlined text-[15px] text-[#06b6d4]">file_download</span>
          <span>Export</span>
        </button>

        {/* Shortcuts Reference Trigger */}
        <button
          onClick={onOpenShortcuts}
          type="button"
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#141c2e] hover:bg-[#1a2438] text-xs font-medium text-[#a0b4c4] hover:text-[#7dd3fc] border border-[#2a3a48]/40 hover:border-[#7dd3fc]/50 transition-colors cursor-pointer"
          title="Keyboard Shortcuts Reference (?)"
        >
          <span className="material-symbols-outlined text-[16px] text-[#7dd3fc]">keyboard</span>
          <span className="hidden xl:inline text-[11px] font-mono">Shortcuts</span>
        </button>

        {/* User Icon */}
        <div className="w-7 h-7 rounded-full bg-[#06b6d4] text-[#001f2e] flex items-center justify-center font-bold text-xs">
          <span className="material-symbols-outlined text-[16px]">person</span>
        </div>
      </div>
    </header>
  );
};
