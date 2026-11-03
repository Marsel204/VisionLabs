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
  polygon_normalized?: number[][] | null;
  polygon_pixels?: number[][] | null;
  source?: string;
}

export interface ImageMeta {
  filename: string;
  path?: string;
  image_id?: string;
  dataset_id?: string;
  width: number;
  height: number;
  size_bytes: number;
  annotation_count: number;
  difficulty?: number;
  status: 'pending' | 'unreviewed' | 'reviewed' | 'ai_labeled';
}

export interface DatasetStats {
  directory: string;
  total_annotations?: number;
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
  skipped_unreviewed?: number;
}

export interface ModelPreparationStatus {
  backend: 'local' | 'cloud';
  models: Record<string, {
    name: string;
    state: 'queued' | 'preparing' | 'ready' | 'error';
    error: string | null;
  }>;
}

export interface AutoLabelStatus {
  warnings?: string[];
  job_id?: string;
  dataset_id?: string;
  succeeded_count?: number;
  failed_count?: number;
  failures?: Array<{image: string; error: string}>;
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
  inference?: { backend: 'local' | 'cloud' };
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
    locate_anything?: boolean;
  };
}

export type ToolType = 'select' | 'pan' | 'bbox' | 'polygon';

export type Page = 'overview' | 'projects' | 'import' | 'autolabel' | 'review' | 'quality' | 'export';

const swatch = (hex: string, alpha = 0.14) => {
  const n = parseInt(hex.slice(1), 16);
  return { stroke: hex, bg: `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`, text: hex };
};

export const CLASS_COLORS: Record<string, { stroke: string; bg: string; text: string }> = {
  object: swatch('#2f6fe4'),
  motorcycle: swatch('#2f6fe4'),
  car: swatch('#1d8fd1'),
  rider: swatch('#8b7be0'),
  person: swatch('#8b7be0'),
  minivan: swatch('#b05fc9'),
  bus: swatch('#16935a'),
  truck: swatch('#0f8a84'),
};

/** Boxes below this confidence are drawn in the warning colour, matching the design. */
export const LOW_CONFIDENCE = 0.65;
export const LOW_CONFIDENCE_COLOR = swatch('#d9822b', 0.12);

const PALETTE = ['#2f6fe4', '#8b7be0', '#16935a', '#1d8fd1', '#c2508e', '#0f8a84', '#b05fc9', '#5b6ee8', '#a8781a']
  .map(hex => swatch(hex));

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

export type AuditKind =
  | 'possibly_missed' | 'misaligned' | 'class_mismatch'
  | 'duplicate' | 'unsupported' | 'implausible_shape';

export interface AuditBox {
  norm_left: number; norm_top: number; norm_right: number; norm_bottom: number;
  x: number; y: number; width: number; height: number;
}

export interface AuditFinding {
  id: string;
  kind: AuditKind;
  severity: number;
  message: string;
  annotation_id: string | null;
  class_name: string | null;
  confidence: number | null;
  occlusion: number;
  likely_occluded: boolean;
  box: AuditBox | null;
}

export interface AuditOcclusion {
  annotation_id: string;
  fraction: number;
  occluded: boolean;
  occluder_ids: string[];
}

export interface AuditReport {
  image_name: string;
  width: number;
  height: number;
  findings: AuditFinding[];
  occlusion: AuditOcclusion[];
  summary: Partial<Record<AuditKind, number>>;
  prediction_count: number;
}
