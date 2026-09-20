import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import type { ImageMeta } from '../../types';
import { getImageUrl } from '../../services/api';

interface Props {
  images: ImageMeta[];
  currentImageIndex: number;
  onSelectImage: (index: number) => void;
}

const ITEM_HEIGHT = 64; // px per thumbnail row
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
    <aside className="w-[215px] flex-shrink-0 bg-[#0f1524]/80 backdrop-blur-xl border-r border-[#2a3a48]/40 flex flex-col z-20 shadow-lg overflow-hidden select-none">
      {/* Header with Search & Filter */}
      <div className="p-2 bg-[#111828]/80 flex flex-col gap-1.5 border-b border-[#2a3a48]/30">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-semibold tracking-wide uppercase text-[#e0e8f0]">Image Dataset</span>
          <span className="text-[10px] font-mono text-[#06b6d4] font-medium">
            {filteredImages.length}/{images.length}
          </span>
        </div>

        {/* Search input */}
        <div className="relative">
          <span className="material-symbols-outlined absolute left-1.5 top-1/2 -translate-y-1/2 text-[#a0b4c4] text-[13px]">
            search
          </span>
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Filter images..."
            className="w-full bg-[#0a0e1a] text-[10px] text-[#e0e8f0] pl-5 pr-2 py-1 rounded focus:outline-none focus:ring-1 focus:ring-[#06b6d4] placeholder:text-[#a0b4c4]/40 font-mono border border-[#2a3a48]/40"
          />
        </div>

        {/* Status Filter Pills */}
        <div className="flex items-center gap-1 text-[8.5px] font-mono overflow-x-auto pb-0.5">
          {(['all', 'unreviewed', 'ai_labeled', 'reviewed'] as const).map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setStatusFilter(f)}
              className={`px-1.5 py-0.5 rounded cursor-pointer transition-colors whitespace-nowrap ${
                statusFilter === f
                  ? f === 'all'
                    ? 'bg-[#0e4d6e] text-[#7dd3fc] font-semibold'
                    : f === 'unreviewed'
                    ? 'bg-[#f59e0b]/20 text-[#fbbf24] font-semibold'
                    : f === 'ai_labeled'
                    ? 'bg-[#a855f7]/20 text-[#c084fc] font-semibold'
                    : 'bg-[#10b981]/20 text-[#34d399] font-semibold'
                  : 'bg-[#141c2e] text-[#a0b4c4] hover:text-[#e0e8f0]'
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
          className={`flex items-center justify-between px-2 py-1 rounded text-[9px] font-mono transition-colors border cursor-pointer ${
            sortByDifficulty
              ? 'bg-[#ef4444]/20 text-[#fca5a5] border-[#ef4444]/50'
              : 'bg-[#141c2e] text-[#a0b4c4] border-[#2a3a48]/40 hover:text-[#e0e8f0]'
          }`}
          title="Triage dataset by Active Learning uncertainty score"
        >
          <div className="flex items-center gap-1">
            <span className="material-symbols-outlined text-[12px]">bolt</span>
            <span>Active Learning Queue</span>
          </div>
          <span className="text-[8px] uppercase">{sortByDifficulty ? 'ON' : 'OFF'}</span>
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
                    className={`p-1.5 rounded-lg transition-all cursor-pointer flex items-center gap-1.5 ${
                      isSelected
                        ? 'bg-[#1a2438] ring-1 ring-[#06b6d4]/70 shadow-sm'
                        : 'bg-[#141c2e]/60 hover:bg-[#1a2438]/60 border border-[#2a3a48]/20'
                    }`}
                  >
                    {/* Thumbnail */}
                    <div className="relative w-10 h-10 flex-shrink-0 rounded overflow-hidden bg-[#0a0e1a] border border-[#2a3a48]/30">
                      <img
                        src={getImageUrl(img.filename)}
                        alt={img.filename}
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                      <div className="absolute bottom-0 right-0 px-0.5 rounded-tl text-[7px] font-mono bg-[#06b6d4] text-[#001f2e] font-bold">
                        JPG
                      </div>
                    </div>

                    {/* Text Info */}
                    <div className="flex-1 min-w-0 flex flex-col justify-between h-10">
                      <span
                        className={`font-mono text-[10px] truncate font-medium ${
                          isSelected ? 'text-[#e0e8f0]' : 'text-[#a0b4c4]'
                        }`}
                        title={img.filename}
                      >
                        {img.filename}
                      </span>
                      <div className="flex items-center gap-1 text-[8px] font-mono">
                        <span className="text-[#a0b4c4]/70 truncate">
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
                          <span className="px-1 rounded text-[7.5px] font-mono bg-[#10b981]/20 text-[#34d399] font-semibold">
                            {img.annotation_count} rev
                          </span>
                        ) : status === 'ai_labeled' ? (
                          <span className="px-1 rounded text-[7.5px] font-mono bg-[#a855f7]/20 text-[#c084fc] font-semibold">
                            {img.annotation_count} AI
                          </span>
                        ) : (
                          <span className="px-1 rounded text-[7.5px] font-mono bg-[#f59e0b]/20 text-[#fbbf24] font-medium">
                            Unreviewed
                          </span>
                        )}
                        <span className="text-[7.5px] font-mono text-[#4a6070]">
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
      <div className="p-1.5 bg-[#111828]/90 border-t border-[#2a3a48]/30 flex items-center justify-between text-[10px] font-mono text-[#a0b4c4]">
        <span>
          {currentImageIndex + 1} / {images.length}
        </span>
        <div className="flex items-center gap-1">
          <button
            onClick={() => onSelectImage(Math.max(0, currentImageIndex - 1))}
            disabled={currentImageIndex === 0}
            className="w-5 h-5 rounded bg-[#141c2e] hover:bg-[#1a2438] flex items-center justify-center disabled:opacity-30 cursor-pointer"
          >
            <span className="material-symbols-outlined text-[13px]">chevron_left</span>
          </button>
          <button
            onClick={() => onSelectImage(Math.min(images.length - 1, currentImageIndex + 1))}
            disabled={currentImageIndex >= images.length - 1}
            className="w-5 h-5 rounded bg-[#141c2e] hover:bg-[#1a2438] flex items-center justify-center disabled:opacity-30 cursor-pointer"
          >
            <span className="material-symbols-outlined text-[13px]">chevron_right</span>
          </button>
        </div>
      </div>
    </aside>
  );
};
