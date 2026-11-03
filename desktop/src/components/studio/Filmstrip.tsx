import React from 'react';
import type { ImageMeta } from '../../types';
import { getImageUrl, imageIdentity } from '../../services/api';

interface Props {
  images: ImageMeta[];
  currentIndex: number;
  onSelect: (index: number) => void;
}

/** Thumbnails around the current image. Only a window is rendered so large datasets stay fast. */
const WINDOW = 14;

export const Filmstrip: React.FC<Props> = ({ images, currentIndex, onSelect }) => {
  const start = Math.max(0, Math.min(currentIndex - WINDOW, images.length - WINDOW * 2 - 1));
  const visible = images.slice(start, start + WINDOW * 2 + 1);
  return (
    <div className="flex h-[60px] shrink-0 items-center gap-3 border-t border-line px-3">
      <span className="text-[11px] text-ink-500">Dataset</span>
      <div className="flex min-w-0 flex-1 gap-1.5 overflow-hidden" role="listbox" aria-label="Gambar dalam dataset">
        {visible.map((image, offset) => {
          const index = start + offset;
          const active = index === currentIndex;
          const dot = image.status === 'reviewed' ? 'bg-brand-600' : image.status === 'ai_labeled' ? 'bg-warn-500' : null;
          return (
            <button
              key={imageIdentity(image)}
              type="button"
              role="option"
              aria-selected={active}
              title={`${image.filename}${image.status === 'reviewed' ? ' · disetujui' : ''}`}
              onClick={() => onSelect(index)}
              className={`relative h-10 w-[54px] shrink-0 overflow-hidden rounded-md cursor-pointer ${
                active ? 'ring-2 ring-brand-600 ring-offset-1' : 'opacity-80 hover:opacity-100'
              }`}
            >
              <img src={getImageUrl(imageIdentity(image))} alt="" loading="lazy" className="h-full w-full object-cover" />
              {dot && <span className={`absolute right-1 top-1 h-1.5 w-1.5 rounded-full ring-1 ring-white ${dot}`} />}
            </button>
          );
        })}
      </div>
      <span className="text-[11px] tabular-nums text-ink-500">{images.length ? currentIndex + 1 : 0} / {images.length}</span>
    </div>
  );
};
