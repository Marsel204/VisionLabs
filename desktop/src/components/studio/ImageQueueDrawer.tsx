import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import type { ImageMeta } from '../../types';
import { getImageUrl } from '../../services/api';

interface Props {
  images: ImageMeta[];
  currentImageIndex: number;
  onSelectImage: (index: number) => void;
}

const ITEM_HEIGHT = 68; // px per thumbnail row
const OVERSCAN = 5;     // extra items above/below visible window

export const ImageQueueDrawer: React.FC<Props> = ({
  images,
  currentImageIndex,
  onSelectImage,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'unreviewed' | 'ai_labeled' | 'reviewed'>('all');
  const [sortByDifficulty, setSortByDifficulty] = useState<boolean>(false);
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(400);
  const listRef = useRef<HTMLDivElement>(null);

  // Measure container height on mount and resize
  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    setContainerHeight(el.clientHeight);
    const ro = new ResizeObserver(() => setContainerHeight(el.clientHeight));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const filteredImages = useMemo(
    () =>
      images
        .map((img, originalIndex) => ({ img, originalIndex }))
        .filter(({ img }) => {
          const matchesSearch = img.filename.toLowerCase().includes(searchTerm.toLowerCase());
          if (statusFilter === 'all') return matchesSearch;
          const st = img.status || (img.annotation_count > 0 ? 'reviewed' : 'unreviewed');
          return matchesSearch && st === statusFilter;
        })
        .sort((a, b) => {
          if (sortByDifficulty) {
            return (b.img.difficulty || 0) - (a.img.difficulty || 0);
          }
          return 0;
        }),
    [images, searchTerm, statusFilter, sortByDifficulty]
  );

  // Auto-scroll to keep selected image visible
  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    const selectedFiltered = filteredImages.findIndex((f) => f.originalIndex === currentImageIndex);
    if (selectedFiltered < 0) return;
    const itemTop = selectedFiltered * ITEM_HEIGHT;
    const st = el.scrollTop;
    const ch = el.clientHeight;
    if (itemTop < st || itemTop + ITEM_HEIGHT > st + ch) {
      el.scrollTop = Math.max(0, itemTop - ch / 2 + ITEM_HEIGHT / 2);
    }
  }, [currentImageIndex, filteredImages]);

  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);

  // Virtual window computation
  const totalHeight = filteredImages.length * ITEM_HEIGHT;
  const startIdx = Math.max(0, Math.floor(scrollTop / ITEM_HEIGHT) - OVERSCAN);
  const endIdx = Math.min(
    filteredImages.length,
    Math.ceil((scrollTop + containerHeight) / ITEM_HEIGHT) + OVERSCAN
  );
  const visibleItems = filteredImages.slice(startIdx, endIdx);
  const paddingTop = startIdx * ITEM_HEIGHT;

  return (
    <aside className="w-[245px] flex-shrink-0 bg-[#0f1524]/85 backdrop-blur-xl border-r border-[#2a3a48]/40 flex flex-col z-20 shadow-lg overflow-hidden select-none">
      {/* Header with Search & Filter */}
      <div className="p-2.5 bg-[#111828]/90 flex flex-col gap-2 border-b border-[#2a3a48]/30">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold tracking-wide uppercase text-[#e0e8f0]">Image Dataset</span>
          <span className="text-xs font-mono text-[#06b6d4] font-semibold">
            {filteredImages.length}/{images.length}
          </span>
        </div>

        {/* Search input with generous left padding so icon never overlaps text */}
        <div className="relative">
          <span className="material-symbols-outlined absolute left-2.5 top-1/2 -translate-y-1/2 text-[#94a3b8] text-[16px] pointer-events-none">
            search
          </span>
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Filter images..."
            className="w-full bg-[#0a0e1a] text-xs text-[#f1f5f9] pl-8 pr-2.5 py-1.5 rounded-md focus:outline-none focus:ring-1 focus:ring-[#06b6d4] placeholder:text-[#94a3b8] border border-[#2a3a48]/50 transition-all font-sans"
          />
        </div>

        {/* Status Filter Pills */}
        <div className="grid grid-cols-4 gap-1 text-[11px] font-medium">
          {(['all', 'unreviewed', 'ai_labeled', 'reviewed'] as const).map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setStatusFilter(f)}
              className={`py-1 rounded-md cursor-pointer transition-colors text-center font-medium ${
                statusFilter === f
                  ? f === 'all'
                    ? 'bg-[#0e4d6e] text-[#7dd3fc] font-semibold border border-[#0284c7]/50'
                    : f === 'unreviewed'
                    ? 'bg-[#f59e0b]/20 text-[#fbbf24] font-semibold border border-[#f59e0b]/40'
                    : f === 'ai_labeled'
                    ? 'bg-[#a855f7]/20 text-[#c084fc] font-semibold border border-[#a855f7]/40'
                    : 'bg-[#10b981]/20 text-[#34d399] font-semibold border border-[#10b981]/40'
                  : 'bg-[#141c2e] text-[#cbd5e1] hover:text-white border border-[#2a3a48]/40'
              }`}
            >
              {f === 'all' ? 'All' : f === 'unreviewed' ? 'Unrev' : f === 'ai_labeled' ? 'AI' : 'Done'}
            </button>
          ))}
        </div>

        {/* Active Learning Sort Toggle */}
        <button
          type="button"
          onClick={() => setSortByDifficulty(!sortByDifficulty)}
          className={`flex items-center justify-between px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors border cursor-pointer ${
            sortByDifficulty
              ? 'bg-[#ef4444]/20 text-[#fca5a5] border-[#ef4444]/50'
              : 'bg-[#141c2e] text-[#cbd5e1] border-[#2a3a48]/40 hover:text-white'
          }`}
          title="Triage dataset by Active Learning uncertainty score"
        >
          <div className="flex items-center gap-1.5">
            <span className="material-symbols-outlined text-[15px] text-[#f59e0b]">bolt</span>
            <span>Active Learning Queue</span>
          </div>
          <span className="text-[10px] font-mono font-bold uppercase">{sortByDifficulty ? 'ON' : 'OFF'}</span>
        </button>
      </div>

      {/* Virtualized Image Items List — only visible rows rendered */}
      <div
        ref={listRef}
        className="flex-1 overflow-y-auto"
        onScroll={handleScroll}
      >
        <div style={{ height: totalHeight, position: 'relative' }}>
          <div style={{ position: 'absolute', top: paddingTop, left: 0, right: 0 }}>
            <div className="px-1.5 space-y-1.5 py-1.5">
              {visibleItems.map(({ img, originalIndex }) => {
                const isSelected = originalIndex === currentImageIndex;
                const status = img.status || (img.annotation_count > 0 ? 'reviewed' : 'unreviewed');

                return (
                  <div
                    key={`${img.filename}-${originalIndex}`}
                    onClick={() => onSelectImage(originalIndex)}
                    className={`p-1.5 rounded-lg transition-all cursor-pointer flex items-center gap-2 ${
                      isSelected
                        ? 'bg-[#1a2438] ring-1 ring-[#06b6d4]/70 shadow-sm'
                        : 'bg-[#141c2e]/60 hover:bg-[#1a2438]/60 border border-[#2a3a48]/20'
                    }`}
                  >
                    {/* Thumbnail */}
                    <div className="relative w-11 h-11 flex-shrink-0 rounded overflow-hidden bg-[#0a0e1a] border border-[#2a3a48]/30">
                      <img
                        src={getImageUrl(img.filename)}
                        alt={img.filename}
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                      <div className="absolute bottom-0 right-0 px-1 py-0.2 rounded-tl text-[8px] font-mono bg-[#06b6d4] text-[#001f2e] font-bold">
                        JPG
                      </div>
                    </div>

                    {/* Text Info */}
                    <div className="flex-1 min-w-0 flex flex-col justify-between h-11">
                      <span
                        className={`font-mono text-xs truncate font-medium ${
                          isSelected ? 'text-[#e0e8f0]' : 'text-[#cbd5e1]'
                        }`}
                        title={img.filename}
                      >
                        {img.filename}
                      </span>
                      <div className="flex items-center gap-1.5 text-[10px] font-mono">
                        <span className="text-[#94a3b8] truncate">
                          {img.width}×{img.height}
                        </span>
                        {typeof img.difficulty === 'number' && img.difficulty > 0 && (
                          <span
                            className="px-1 rounded bg-[#ef4444]/20 text-[#fca5a5] font-semibold"
                            title="Active Learning Difficulty"
                          >
                            diff: {img.difficulty.toFixed(2)}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center justify-between">
                        {status === 'reviewed' ? (
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-[#10b981]/20 text-[#34d399] font-semibold">
                            {img.annotation_count} rev
                          </span>
                        ) : status === 'ai_labeled' ? (
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-[#a855f7]/20 text-[#c084fc] font-semibold">
                            {img.annotation_count} AI
                          </span>
                        ) : (
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-[#f59e0b]/20 text-[#fbbf24] font-medium">
                            Unreviewed
                          </span>
                        )}
                        <span className="text-[9px] font-mono text-[#64748b]">
                          #{originalIndex + 1}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Pagination Counter */}
      <div className="p-2 bg-[#111828]/95 border-t border-[#2a3a48]/30 flex items-center justify-between text-xs font-mono text-[#cbd5e1]">
        <span>
          {currentImageIndex + 1} / {images.length}
        </span>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => onSelectImage(Math.max(0, currentImageIndex - 1))}
            disabled={currentImageIndex === 0}
            className="w-6 h-6 rounded bg-[#141c2e] hover:bg-[#1a2438] flex items-center justify-center disabled:opacity-30 cursor-pointer text-[#e0e8f0]"
          >
            <span className="material-symbols-outlined text-[15px]">chevron_left</span>
          </button>
          <button
            onClick={() => onSelectImage(Math.min(images.length - 1, currentImageIndex + 1))}
            disabled={currentImageIndex >= images.length - 1}
            className="w-6 h-6 rounded bg-[#141c2e] hover:bg-[#1a2438] flex items-center justify-center disabled:opacity-30 cursor-pointer text-[#e0e8f0]"
          >
            <span className="material-symbols-outlined text-[15px]">chevron_right</span>
          </button>
        </div>
      </div>
    </aside>
  );
};
