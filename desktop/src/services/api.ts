import type { BoundingBox, ImageMeta, SystemHealth } from '../types';

const BASE_URL = 'http://127.0.0.1:8765';

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch(`${BASE_URL}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchImages(
  limit = 200,
  offset = 0,
  status?: string,
): Promise<{ images: ImageMeta[]; total: number; directory: string }> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (status) params.set('status', status);
  const res = await fetch(`${BASE_URL}/api/images?${params}`);
  if (!res.ok) throw new Error(`Failed to fetch images: ${res.statusText}`);
  return res.json();
}

/**
 * Load ALL images across pagination boundaries.
 * Fetches first page, then remaining pages in parallel 1000-item chunks.
 */
export async function fetchAllImages(): Promise<{ images: ImageMeta[]; total: number; directory: string }> {
  const first = await fetchImages(1000, 0);
  const total = first.total;
  if (total <= 1000) return first;

  // Remaining pages in parallel
  const pageSize = 1000;
  const remaining = Math.ceil((total - 1000) / pageSize);
  const pages = await Promise.allSettled(
    Array.from({ length: remaining }, (_, i) => fetchImages(pageSize, 1000 + i * pageSize)),
  );
  const allImages = [...first.images];
  for (const page of pages) {
    if (page.status === 'fulfilled') allImages.push(...page.value.images);
  }
  return { images: allImages, total, directory: first.directory };
}


export function getImageUrl(filename: string): string {
  return `${BASE_URL}/api/image/${encodeURIComponent(filename)}`;
}

export async function fetchAnnotations(filename: string): Promise<{ boxes: BoundingBox[] }> {
  const res = await fetch(`${BASE_URL}/api/annotations/${encodeURIComponent(filename)}`);
  if (!res.ok) throw new Error(`Failed to fetch annotations: ${res.statusText}`);
  return res.json();
}

export async function saveAnnotations(filename: string, boxes: BoundingBox[]): Promise<void> {
  const res = await fetch(`${BASE_URL}/api/annotations/${encodeURIComponent(filename)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_name: filename, boxes }),
  });
  if (!res.ok) throw new Error(`Failed to save annotations: ${res.statusText}`);
}

export async function detectYolo(filename: string, confThreshold: number = 0.25, models?: string[]): Promise<{ boxes: BoundingBox[]; width: number; height: number }> {
  const payload: Record<string, any> = { image_name: filename, conf_threshold: confThreshold };
  if (models && models.length > 0) {
    payload.models = models;
  }
  const res = await fetch(`${BASE_URL}/api/detect/yolo`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`YOLO detection failed: ${res.statusText}`);
  return res.json();
}

export async function autoRefinePrompt(className: string, currentPrompt?: string): Promise<{ refined_prompt: string; suggestions: string[] }> {
  const res = await fetch(`${BASE_URL}/api/prompt/auto-refine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ class_name: className, current_prompt: currentPrompt }),
  });
  if (!res.ok) throw new Error(`Prompt refinement failed: ${res.statusText}`);
  return res.json();
}

export async function fetchDatasetStats(): Promise<import('../types').DatasetStats> {
  const res = await fetch(`${BASE_URL}/api/dataset/stats`);
  if (!res.ok) throw new Error(`Failed to fetch dataset stats: ${res.statusText}`);
  return res.json();
}

export async function runAutoLabelPreview(payload: {
  image_name?: string;
  classes: Array<{ name: string; prompt: string; color: string; enabled: boolean }>;
  confidence_threshold?: number;
  iou_threshold?: number;
  strict_vlm?: boolean;
  max_instances?: number;
  pipeline_mode?: string;
  enable_grounding_dino?: boolean;
  enable_sam2_masks?: boolean;
  enable_florence2?: boolean;
  enable_florence2_verifier?: boolean;
  enable_yolo?: boolean;
  yolo_models?: string[];
}): Promise<any> {
  const res = await fetch(`${BASE_URL}/api/autolabel/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Auto-label preview failed: ${res.statusText}`);
  return res.json();
}

export async function startAutoLabelBatch(payload: {
  classes: Array<{ name: string; prompt: string; color: string; enabled: boolean }>;
  confidence_threshold?: number;
  iou_threshold?: number;
  pipeline_mode?: string;
  only_unannotated?: boolean;
  enable_grounding_dino?: boolean;
  enable_sam2_masks?: boolean;
  enable_florence2?: boolean;
  enable_florence2_verifier?: boolean;
  enable_yolo?: boolean;
  yolo_models?: string[];
}): Promise<{ status: string; total: number; classes: string[] }> {
  const res = await fetch(`${BASE_URL}/api/autolabel/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Batch auto-labeling failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModelPresets(): Promise<{ presets: Array<{ id: string; name: string; size: string; type: string }> }> {
  const res = await fetch(`${BASE_URL}/api/models/presets`);
  if (!res.ok) throw new Error(`Failed to fetch model presets: ${res.statusText}`);
  return res.json();
}

export async function browseCustomWeights(): Promise<{ status: string; path?: string; name?: string }> {
  const res = await fetch(`${BASE_URL}/api/models/browse-weights`, { method: 'POST' });
  if (!res.ok) throw new Error(`Failed to browse weights: ${res.statusText}`);
  return res.json();
}

export async function validateCustomModel(path: string): Promise<{ status: string; name: string; path: string; classes: string[] }> {
  const res = await fetch(`${BASE_URL}/api/models/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  });
  if (!res.ok) throw new Error(`Failed to validate model: ${res.statusText}`);
  return res.json();
}

export async function fetchAutoLabelStatus(): Promise<import('../types').AutoLabelStatus> {
  const res = await fetch(`${BASE_URL}/api/autolabel/status`);
  if (!res.ok) throw new Error(`Failed to fetch auto-label status: ${res.statusText}`);
  return res.json();
}

export async function exportDataset(payload: {
  format: 'yolo' | 'coco';
  train_ratio?: number;
  val_ratio?: number;
  test_ratio?: number;
  output_dir?: string;
}): Promise<import('../types').ExportResult> {
  const res = await fetch(`${BASE_URL}/api/dataset/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Dataset export failed: ${res.statusText}`);
  return res.json();
}

export async function fetchActiveLearningQueue(limit: number = 20): Promise<{ queue: any[]; count: number }> {
  const res = await fetch(`${BASE_URL}/api/active-learning/queue?limit=${limit}`);
  if (!res.ok) throw new Error(`Failed to fetch active learning queue: ${res.statusText}`);
  return res.json();
}

export async function selectDatasetFolder(folderPath: string): Promise<{ status: string; active_directory: string; indexed_count: number }> {
  const res = await fetch(`${BASE_URL}/api/dataset/select-folder?path=${encodeURIComponent(folderPath)}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Failed to switch dataset folder: ${res.statusText}`);
  return res.json();
}

export async function rescanDataset(): Promise<{ status: string; indexed_count: number; directory: string }> {
  const res = await fetch(`${BASE_URL}/api/dataset/rescan`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Failed to rescan dataset: ${res.statusText}`);
  return res.json();
}

export async function browseDatasetFolder(): Promise<{ status: string; path?: string; active_directory?: string; indexed_count?: number }> {
  const res = await fetch(`${BASE_URL}/api/dataset/browse-folder`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Failed to browse folder: ${res.statusText}`);
  return res.json();
}

export async function searchDatasetDirectories(query: string = ''): Promise<{ directories: string[] }> {
  const res = await fetch(`${BASE_URL}/api/dataset/search-directories?query=${encodeURIComponent(query)}`);
  if (!res.ok) throw new Error(`Failed to search directories: ${res.statusText}`);
  return res.json();
}

export async function uploadDatasetImages(files: FileList | File[]): Promise<{ saved: string[]; count: number; total_indexed: number }> {
  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
  }
  const res = await fetch(`${BASE_URL}/api/dataset/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`Failed to upload images: ${res.statusText}`);
  return res.json();
}
