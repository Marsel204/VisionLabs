export interface BoundingBox {
  id: string;
  class_name: string;
  class_id: number;
  confidence: number;
  x: number; // pixel x top-left
  y: number; // pixel y top-left
  width: number;
  height: number;
  norm_left: number;
  norm_top: number;
  norm_right: number;
  norm_bottom: number;
  occluded?: boolean;
  truncated?: boolean;
  difficult_lighting?: boolean;
  source?: string;
}

export interface ImageMeta {
  filename: string;
  path?: string;
  width: number;
  height: number;
  size_bytes: number;
  annotation_count: number;
  difficulty?: number;
  status: 'pending' | 'reviewed' | 'ai_labeled';
}

export interface DatasetStats {
  directory: string;
  total_images: number;
  unreviewed: number;
  reviewed: number;
  ai_labeled: number;
  class_counts: Record<string, number>;
  database_file: string;
}

export interface ExportResult {
  status: string;
  format: string;
  destination: string;
  artifact: string;
  splits: {
    train: number;
    val: number;
    test: number;
  };
  total_documents: number;
}

export interface AutoLabelStatus {
  running: boolean;
  current: number;
  total: number;
  current_image: string;
  completed: boolean;
  error: string | null;
  processed_count: number;
}


export interface SystemHealth {
  status: string;
  dataset_dir: string;
  gpu: {
    available: boolean;
    device: string;
    vram_free: string;
    temperature?: string;
  };
  classes: string[];
  models: {
    yolo11: boolean;
    sam2: boolean;
    grounding_dino: boolean;
    florence2: boolean;
  };
}

export type ToolType = 'select' | 'pan' | 'bbox' | 'polygon';

export const CLASS_COLORS: Record<string, { stroke: string; bg: string; text: string }> = {
  object: { stroke: '#06b6d4', bg: 'rgba(6, 182, 212, 0.15)', text: '#06b6d4' },
  car: { stroke: '#06b6d4', bg: 'rgba(6, 182, 212, 0.15)', text: '#06b6d4' },
  motorcycle: { stroke: '#f59e0b', bg: 'rgba(245, 158, 11, 0.18)', text: '#f59e0b' },
  minivan: { stroke: '#a855f7', bg: 'rgba(168, 85, 247, 0.18)', text: '#c084fc' },
  bus: { stroke: '#10b981', bg: 'rgba(16, 185, 129, 0.18)', text: '#34d399' },
  truck: { stroke: '#3b82f6', bg: 'rgba(59, 130, 246, 0.18)', text: '#60a5fa' },
  person: { stroke: '#ec4899', bg: 'rgba(236, 72, 153, 0.18)', text: '#f472b6' },
};

const PALETTE = [
  { stroke: '#06b6d4', bg: 'rgba(6, 182, 212, 0.15)', text: '#06b6d4' },
  { stroke: '#f59e0b', bg: 'rgba(245, 158, 11, 0.18)', text: '#f59e0b' },
  { stroke: '#10b981', bg: 'rgba(16, 185, 129, 0.18)', text: '#34d399' },
  { stroke: '#ec4899', bg: 'rgba(236, 72, 153, 0.18)', text: '#f472b6' },
  { stroke: '#3b82f6', bg: 'rgba(59, 130, 246, 0.18)', text: '#60a5fa' },
  { stroke: '#a855f7', bg: 'rgba(168, 85, 247, 0.18)', text: '#c084fc' },
  { stroke: '#eab308', bg: 'rgba(234, 179, 8, 0.18)', text: '#fde047' },
  { stroke: '#14b8a6', bg: 'rgba(20, 184, 166, 0.18)', text: '#2dd4bf' },
  { stroke: '#f97316', bg: 'rgba(249, 115, 22, 0.18)', text: '#fb923c' },
];

export function getClassColor(className: string): { stroke: string; bg: string; text: string } {
  const lower = (className || 'object').toLowerCase();
  if (CLASS_COLORS[lower]) {
    return CLASS_COLORS[lower];
  }
  let hash = 0;
  for (let i = 0; i < lower.length; i++) {
    hash = (hash << 5) - hash + lower.charCodeAt(i);
    hash |= 0;
  }
  const index = Math.abs(hash) % PALETTE.length;
  return PALETTE[index];
}
