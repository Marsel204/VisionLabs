import React from 'react';
import type { ToolType } from '../../types';

interface Props {
  activeTool: ToolType;
  onSelectTool: (tool: ToolType) => void;
}

export const ActivityRail: React.FC<Props> = ({ activeTool, onSelectTool }) => {
  const tools: { type: ToolType; icon: string; title: string; shortcut: string }[] = [
    { type: 'select', icon: 'near_me', title: 'Select & Transform', shortcut: 'V' },
    { type: 'pan', icon: 'pan_tool', title: 'Pan Canvas', shortcut: 'H' },
    { type: 'bbox', icon: 'crop_square', title: '2D Bounding Box', shortcut: 'B' },
    { type: 'polygon', icon: 'polyline', title: 'Polygon Segmentation', shortcut: 'P' },
  ];

  return (
    <aside className="w-11 flex-shrink-0 bg-[#0f1524]/90 backdrop-blur-xl border-r border-[#2a3a48]/40 flex flex-col items-center py-2.5 justify-between select-none z-30">
      {/* Top Navigation Tool Icons */}
      <nav className="flex flex-col items-center gap-1.5">
        {tools.map((t) => {
          const isActive = activeTool === t.type;
          return (
            <button
              key={t.type}
              onClick={() => onSelectTool(t.type)}
              type="button"
              title={`${t.title} (${t.shortcut})`}
              className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all cursor-pointer ${
                isActive
                  ? 'bg-[#0e4d6e] text-[#7dd3fc] border border-[#7dd3fc]/50 shadow-sm'
                  : 'bg-[#141c2e]/60 text-[#a0b4c4] border border-transparent hover:border-[#2a3a48] hover:bg-[#1a2438] hover:text-[#e0e8f0]'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">{t.icon}</span>
            </button>
          );
        })}
      </nav>

      {/* Bottom Settings Button */}
      <div className="flex flex-col items-center gap-1.5">
        <button
          type="button"
          title="Studio Settings"
          className="w-8 h-8 rounded-lg flex items-center justify-center text-[#a0b4c4] hover:bg-[#1a2438] hover:text-[#e0e8f0] transition-colors cursor-pointer"
        >
          <span className="material-symbols-outlined text-[16px]">settings</span>
        </button>
      </div>
    </aside>
  );
};
