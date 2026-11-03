import type { BoundingBox, ImageMeta } from './types';

export type ReviewStatus = 'reviewed' | 'unreviewed' | 'ai_labeled';

/** Editing a box is not approving the image: keep whatever review state the image already has. */
export function statusAfterEdit(status: ImageMeta['status'] | undefined): ReviewStatus {
  return status === 'reviewed' || status === 'ai_labeled' ? status : 'unreviewed';
}

function overlap(a: BoundingBox, b: BoundingBox) {
  const width = Math.min(a.norm_right, b.norm_right) - Math.max(a.norm_left, b.norm_left);
  const height = Math.min(a.norm_bottom, b.norm_bottom) - Math.max(a.norm_top, b.norm_top);
  if (width <= 0 || height <= 0) return { iou: 0, containment: 0, ratio: 0 };
  const inter = width * height;
  const areaA = (a.norm_right - a.norm_left) * (a.norm_bottom - a.norm_top);
  const areaB = (b.norm_right - b.norm_left) * (b.norm_bottom - b.norm_top);
  return {
    iou: inter / (areaA + areaB - inter),
    containment: inter / Math.min(areaA, areaB),
    ratio: Math.min(areaA, areaB) / Math.max(areaA, areaB),
  };
}

/** Whether a detection just repeats a box that is already labelled. */
export function isAlreadyLabelled(detection: BoundingBox, existing: BoundingBox[]): boolean {
  return existing.some(box => {
    const { iou, containment, ratio } = overlap(box, detection);
    const sameClass = box.class_name.toLowerCase() === detection.class_name.toLowerCase();
    // A smaller box inside a larger one (a hidden neighbour) is a different object, so
    // containment only counts when the two are about the same size.
    return (sameClass && (iou >= 0.5 || (containment >= 0.9 && ratio >= 0.6))) || iou >= 0.7;
  });
}

export interface MergeResult {
  boxes: BoundingBox[];
  added: number;
  skipped: number;
}

/**
 * Add detector boxes that are not labelled yet. Existing boxes are never changed or removed,
 * and new boxes get fresh ids so they cannot collide with ones from an earlier detection.
 */
export function mergeDetections(
  existing: BoundingBox[], detections: BoundingBox[], newId: () => string = () => crypto.randomUUID(),
): MergeResult {
  const additions: BoundingBox[] = [];
  let skipped = 0;
  for (const detection of detections) {
    if (isAlreadyLabelled(detection, [...existing, ...additions])) skipped++;
    else additions.push({ ...detection, id: newId() });
  }
  return { boxes: [...existing, ...additions], added: additions.length, skipped };
}

const TEXT_INPUT_TYPES = new Set(['text', 'search', 'number', 'email', 'password', 'url', 'tel', 'date', 'time']);

interface KeyTarget {
  tagName?: string;
  type?: string;
  isContentEditable?: boolean;
  getAttribute?: (name: string) => string | null;
}

/**
 * Elements where a keystroke means something other than a studio shortcut: typing, choosing an
 * option, or moving a slider. Checkboxes and buttons are not on this list, so shortcuts keep
 * working after you tick "Tertutup".
 */
export function isTypingTarget(target: unknown): boolean {
  const el = target as KeyTarget | null;
  if (!el?.tagName) return false;
  if (el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable) return true;
  if (el.tagName === 'INPUT') return TEXT_INPUT_TYPES.has(el.type || 'text') || el.type === 'range';
  return false;
}

/** Enter and Space activate a focused button, link, row or checkbox; they must not also fire a shortcut. */
export function activatesFocusedControl(target: unknown, key: string): boolean {
  if (key !== 'Enter' && key !== ' ') return false;
  const el = target as KeyTarget | null;
  if (!el?.tagName) return false;
  if (['BUTTON', 'A', 'SUMMARY'].includes(el.tagName)) return true;
  if (el.tagName === 'INPUT') return ['checkbox', 'radio'].includes(el.type || '');
  const role = el.getAttribute?.('role');
  return !!role && ['button', 'option', 'tab', 'radio', 'checkbox'].includes(role);
}
