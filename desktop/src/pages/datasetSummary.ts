import type { ImageMeta } from '../types';

export interface DatasetSummary {
  total: number;
  approved: number;
  autoLabeled: number;
  unreviewed: number;
  /** Neither reviewed nor auto-labeled and without any box: what a batch auto-label will process. */
  unannotated: number;
  pending: number;
  annotations: number;
  approvedPercent: number;
}

/** Review progress from the image index. "Approved" means a person saved or approved the image. */
export function summarizeDataset(images: ImageMeta[]): DatasetSummary {
  let approved = 0;
  let autoLabeled = 0;
  let annotations = 0;
  let unannotated = 0;
  for (const image of images) {
    if (image.status === 'reviewed') approved++;
    else if (image.status === 'ai_labeled') autoLabeled++;
    else if (!(image.annotation_count > 0)) unannotated++;
    annotations += Math.max(0, image.annotation_count || 0);
  }
  const total = images.length;
  return {
    total,
    approved,
    autoLabeled,
    unreviewed: total - approved - autoLabeled,
    unannotated,
    pending: total - approved,
    annotations,
    approvedPercent: total ? Math.round((approved / total) * 100) : 0,
  };
}

export function datasetNameFrom(directory: string | undefined): string {
  if (!directory) return '';
  const parts = directory.replace(/[\\/]+$/, '').split(/[\\/]/);
  return parts[parts.length - 1] || directory;
}

/** Index of the first image that still needs a person, or 0. */
export function firstPendingIndex(images: ImageMeta[]): number {
  const index = images.findIndex(image => image.status !== 'reviewed');
  return index < 0 ? 0 : index;
}
