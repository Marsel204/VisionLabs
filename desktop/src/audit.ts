import type { AuditFinding, AuditKind, AuditReport, BoundingBox } from './types';

export const AUDIT_KIND_LABELS: Record<AuditKind, { label: string; icon: string; color: string; action: string }> = {
  possibly_missed: { label: 'Mungkin terlewat', icon: 'add_box', color: '#d9822b', action: 'Tambah kotak' },
  misaligned: { label: 'Posisi meleset', icon: 'open_with', color: '#1d8fd1', action: 'Perbaiki kotak' },
  class_mismatch: { label: 'Kelas salah', icon: 'label_important', color: '#8b7be0', action: 'Ganti kelas' },
  duplicate: { label: 'Duplikat', icon: 'content_copy', color: '#c4493a', action: 'Hapus' },
  unsupported: { label: 'Tanpa bukti', icon: 'help', color: '#c2508e', action: 'Hapus' },
  implausible_shape: { label: 'Bentuk aneh', icon: 'aspect_ratio', color: '#a8781a', action: 'Hapus' },
};

/** Indonesian description of a finding, built from its fields rather than the engine's English text. */
export function describeFinding(finding: AuditFinding): string {
  const confidence = finding.confidence != null ? ` (${Math.round(finding.confidence * 100)}%)` : '';
  const hidden = finding.occlusion >= 0.25 ? `, sekitar ${Math.round(finding.occlusion * 100)}% tertutup` : '';
  switch (finding.kind) {
    case 'possibly_missed':
      return `Kemungkinan ${finding.class_name ?? 'objek'} belum dilabeli${confidence}${hidden}.`;
    case 'misaligned':
      return `Kotak tidak pas dengan ${finding.class_name ?? 'objek'} yang terdeteksi${confidence}.`;
    case 'class_mismatch':
      return `Detektor melihat ${finding.class_name ?? 'kelas lain'}${confidence}, bukan kelas yang dilabeli.`;
    case 'duplicate':
      return `Duplikat dari kotak ${finding.class_name ?? ''} lain di tempat yang sama.`;
    case 'unsupported':
      return `Detektor tidak menemukan objek di kotak ini${hidden ? `${hidden}, jadi bisa saja tetap benar` : ''}.`;
    default:
      return 'Kotak sangat kecil atau perbandingan sisinya ekstrem.';
  }
}

/** Findings that still refer to boxes currently present and have not been dismissed. */
export function visibleFindings(
  report: AuditReport | null, boxes: BoundingBox[], dismissed: ReadonlySet<string>,
): AuditFinding[] {
  if (!report) return [];
  const ids = new Set(boxes.map(box => box.id));
  return report.findings.filter(finding => !dismissed.has(finding.id)
    && (finding.annotation_id === null || ids.has(finding.annotation_id)));
}

function withGeometry(box: BoundingBox, g: NonNullable<AuditFinding['box']>): BoundingBox {
  // Polygons no longer match a re-drawn box, so they are dropped rather than left misaligned.
  return { ...box, x: g.x, y: g.y, width: g.width, height: g.height,
    norm_left: g.norm_left, norm_top: g.norm_top, norm_right: g.norm_right, norm_bottom: g.norm_bottom,
    polygon_normalized: null, polygon_pixels: null };
}

/** Return the box list after applying a finding's suggested fix, or null if it cannot be applied. */
export function applyFinding(
  finding: AuditFinding, boxes: BoundingBox[], classes: string[],
): BoundingBox[] | null {
  const target = finding.annotation_id ? boxes.find(box => box.id === finding.annotation_id) : undefined;
  const classId = finding.class_name
    ? classes.findIndex(name => name.toLowerCase() === finding.class_name!.toLowerCase()) : -1;
  switch (finding.kind) {
    case 'possibly_missed': {
      if (!finding.box || classId < 0) return null;
      const added = withGeometry({
        id: crypto.randomUUID(), class_name: classes[classId], class_id: classId, confidence: 1,
        x: 0, y: 0, width: 0, height: 0, norm_left: 0, norm_top: 0, norm_right: 0, norm_bottom: 0,
        occluded: finding.likely_occluded, source: 'yolo',
      }, finding.box);
      return [...boxes, added];
    }
    case 'misaligned':
      if (!target || !finding.box) return null;
      return boxes.map(box => box.id === target.id ? withGeometry(box, finding.box!) : box);
    case 'class_mismatch':
      if (!target || classId < 0) return null;
      return boxes.map(box => box.id === target.id
        ? { ...box, class_name: classes[classId], class_id: classId } : box);
    default:
      if (!target) return null;
      return boxes.filter(box => box.id !== target.id);
  }
}

/** Mark boxes the audit estimates as occluded; never clears a flag set by a person. */
export function applyOcclusionFlags(boxes: BoundingBox[], report: AuditReport | null): BoundingBox[] | null {
  if (!report) return null;
  const occluded = new Set(report.occlusion.filter(item => item.occluded).map(item => item.annotation_id));
  const changed = boxes.some(box => occluded.has(box.id) && !box.occluded);
  return changed ? boxes.map(box => occluded.has(box.id) ? { ...box, occluded: true } : box) : null;
}

/** Display-only image filters (they never change saved labels or detector input). */
export const ENHANCE_PRESETS = [
  { id: 'none', label: 'Asli' },
  { id: 'sharpen', label: 'Tajamkan' },
  { id: 'contrast', label: 'Kontras' },
  { id: 'rain', label: 'Hujan / buram' },
] as const;
export type EnhancePreset = typeof ENHANCE_PRESETS[number]['id'];

export function nextEnhancePreset(current: string): EnhancePreset {
  const index = ENHANCE_PRESETS.findIndex(preset => preset.id === current);
  return ENHANCE_PRESETS[(index + 1) % ENHANCE_PRESETS.length].id;
}

/** How many layered detector passes the audit runs. More passes find more, but add review work. */
export const AUDIT_DEPTHS = [
  { id: 'fast', label: 'Cepat · gambar asli', views: ['original'] },
  { id: 'thorough', label: 'Teliti · + pass cermin', views: ['original', 'flip'] },
  { id: 'max', label: 'Maksimal · + tajam, kontras', views: ['original', 'flip', 'sharpen', 'contrast'] },
] as const;
export type AuditDepth = typeof AUDIT_DEPTHS[number]['id'];

export function auditViews(depth: string): string[] {
  const match = AUDIT_DEPTHS.find(item => item.id === depth) ?? AUDIT_DEPTHS[1];
  return [...match.views];
}
