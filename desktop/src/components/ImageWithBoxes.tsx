import React from 'react';
import { getClassColor, LOW_CONFIDENCE, LOW_CONFIDENCE_COLOR } from '../types';

export interface PreviewBox {
  class_name: string;
  confidence?: number | null;
  norm_left: number;
  norm_top: number;
  norm_right: number;
  norm_bottom: number;
  polygon_normalized?: number[][] | null;
}

interface Props {
  src: string;
  alt: string;
  width: number;
  height: number;
  boxes: PreviewBox[];
  showMasks?: boolean;
  className?: string;
  imgClassName?: string;
}

/** Read-only image with labelled boxes, drawn like the review canvas. */
export const ImageWithBoxes: React.FC<Props> = ({
  src, alt, width, height, boxes, showMasks = true, className = '', imgClassName = '',
}) => {
  const w = width || 1000;
  const h = height || 1000;
  const font = Math.max(10, Math.round(w / 70));
  return (
    <div className={`relative ${className}`}>
      <svg role="img" aria-label={alt} className={`pointer-events-none block h-auto w-full ${imgClassName}`}
        viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet">
        <image href={src} width={w} height={h} />
        {boxes.map((box, index) => {
          const low = box.confidence != null && box.confidence < LOW_CONFIDENCE;
          const color = low ? LOW_CONFIDENCE_COLOR : getClassColor(box.class_name);
          const x = box.norm_left * w;
          const y = box.norm_top * h;
          const bw = Math.max(2, (box.norm_right - box.norm_left) * w);
          const bh = Math.max(2, (box.norm_bottom - box.norm_top) * h);
          const label = `${box.class_name}${box.confidence != null ? ` ${box.confidence.toFixed(2)}` : ''}`;
          const labelWidth = label.length * font * 0.6 + font;
          return (
            <g key={index}>
              {showMasks && box.polygon_normalized && box.polygon_normalized.length > 2 && (
                <polygon
                  points={box.polygon_normalized.map(([px, py]) => `${px * w},${py * h}`).join(' ')}
                  fill={color.bg} stroke={color.stroke} strokeWidth={Math.max(1, w / 600)}
                />
              )}
              <rect x={x} y={y} width={bw} height={bh} fill="none" stroke={color.stroke} strokeWidth={Math.max(1.5, w / 500)} />
              <rect x={x} y={Math.max(0, y - font * 1.5)} width={labelWidth} height={font * 1.5} fill={color.stroke} />
              <text x={x + font * 0.5} y={Math.max(font * 1.1, y - font * 0.4)} fill="#fff" fontSize={font} fontFamily="Manrope, sans-serif" fontWeight={600}>
                {label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};
