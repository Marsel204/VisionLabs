import React from 'react';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

interface ShortcutItem {
  keys: string[];
  description: string;
  badge?: string;
}

interface ShortcutCategory {
  title: string;
  icon: string;
  items: ShortcutItem[];
}

export const KeyboardShortcutsModal: React.FC<Props> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const categories: ShortcutCategory[] = [
    {
      title: 'Tools & Annotation Mode',
      icon: 'construction',
      items: [
        { keys: ['V'], description: 'Select & Transform tool' },
        { keys: ['B'], description: '2D Bounding Box drawing tool' },
        { keys: ['H'], description: 'Hand / Pan Canvas tool' },
        { keys: ['P'], description: 'Polygon Segmentation tool' },
      ],
    },
    {
      title: 'Image Navigation',
      icon: 'photo_library',
      items: [
        { keys: ['A', 'or', '←', 'or', '['], description: 'Previous image in queue' },
        { keys: ['D', 'or', '→', 'or', ']'], description: 'Next image in queue' },
        { keys: ['Home'], description: 'Jump to first image' },
        { keys: ['End'], description: 'Jump to last image' },
      ],
    },
    {
      title: 'Object Selection & Triage',
      icon: 'fact_check',
      items: [
        { keys: ['Tab'], description: 'Cycle select next bounding box' },
        { keys: ['Shift', 'Tab'], description: 'Cycle select previous bounding box' },
        { keys: ['Del', 'or', 'Backspace'], description: 'Delete selected bounding box' },
        { keys: ['Enter', 'or', 'Space'], description: 'Accept / Validate selected box (100% conf)' },
        { keys: ['Esc'], description: 'Deselect box / Close active modal' },
      ],
    },
    {
      title: 'Quick Class Assignment (1-6)',
      icon: 'label',
      items: [
        { keys: ['1-6'], description: 'Quickly assign corresponding class 1-6 to selected object' },
      ],
    },
    {
      title: 'AI & Automation',
      icon: 'smart_toy',
      items: [
        { keys: ['R'], description: 'Run YOLO AI detection on active image', badge: 'YOLO11' },
        { keys: ['L', 'or', 'M'], description: 'Open Auto-Label AI Configuration & Batch Modal' },
      ],
    },
    {
      title: 'Data & Workstation',
      icon: 'tune',
      items: [
        { keys: ['Ctrl', 'S'], description: 'Save current annotations to disk' },
        { keys: ['I'], description: 'Open Dataset Import dialog' },
        { keys: ['E'], description: 'Open Dataset Export dialog' },
        { keys: ['?'], description: 'Toggle this Shortcuts Guide' },
      ],
    },
  ];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-md animate-fadeIn"
      onClick={onClose}
    >
      <div
        className="w-[720px] max-h-[85vh] bg-[#0c1220] border border-[#2a3a48]/70 rounded-2xl shadow-2xl overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-5 py-3.5 bg-[#0f172a] border-b border-[#2a3a48]/50 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#0e4d6e]/80 border border-[#7dd3fc]/40 flex items-center justify-center text-[#7dd3fc]">
              <span className="material-symbols-outlined text-[19px]">keyboard</span>
            </div>
            <div>
              <h2 className="text-sm font-semibold text-[#e0e8f0] tracking-tight">
                Keyboard Shortcuts Reference
              </h2>
              <p className="text-[11px] text-[#7dd3fc]/80 font-mono">
                VisionLab AI High-Velocity Annotation Hotkeys
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            type="button"
            className="w-7 h-7 rounded-lg bg-[#141c2e] hover:bg-[#1a2438] text-[#a0b4c4] hover:text-[#e0e8f0] flex items-center justify-center transition-colors cursor-pointer"
            title="Close (Esc)"
          >
            <span className="material-symbols-outlined text-[17px]">close</span>
          </button>
        </div>

        {/* Content Grid */}
        <div className="p-5 overflow-y-auto grid grid-cols-1 md:grid-cols-2 gap-4 custom-scrollbar">
          {categories.map((cat, idx) => (
            <div
              key={idx}
              className="bg-[#111828]/60 border border-[#2a3a48]/40 rounded-xl p-3.5 space-y-2.5"
            >
              <div className="flex items-center gap-2 pb-1 border-b border-[#2a3a48]/30">
                <span className="material-symbols-outlined text-[16px] text-[#06b6d4]">
                  {cat.icon}
                </span>
                <span className="text-[11px] font-semibold uppercase tracking-wider text-[#dae2fd]">
                  {cat.title}
                </span>
              </div>

              <div className="space-y-1.5">
                {cat.items.map((item, itemIdx) => (
                  <div
                    key={itemIdx}
                    className="flex items-center justify-between text-[11px] py-0.5"
                  >
                    <span className="text-[#94a3b8] truncate pr-2">{item.description}</span>
                    <div className="flex items-center gap-1 shrink-0">
                      {item.keys.map((k, kIdx) =>
                        k === 'or' ? (
                          <span key={kIdx} className="text-[10px] text-[#64748b] px-0.5">
                            /
                          </span>
                        ) : (
                          <kbd
                            key={kIdx}
                            style={item.badge ? { borderColor: `${item.badge}55` } : {}}
                            className="px-1.5 py-0.5 rounded bg-[#172136] border border-[#334155] text-[10px] font-mono font-semibold text-[#f1f5f9] shadow-sm min-w-[20px] text-center"
                          >
                            {k}
                          </kbd>
                        )
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="px-5 py-2.5 bg-[#0f172a] border-t border-[#2a3a48]/50 flex items-center justify-between text-[11px] font-mono text-[#64748b]">
          <span>Tip: Press <kbd className="px-1 py-0.5 rounded bg-[#172136] text-[#e0e8f0] border border-[#334155]">?</kbd> anytime to toggle this modal</span>
          <button
            onClick={onClose}
            type="button"
            className="px-3 py-1 rounded-md bg-[#0e4d6e] hover:bg-[#0284c7] text-[#e0f2fe] text-xs font-semibold transition-colors cursor-pointer"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
};
