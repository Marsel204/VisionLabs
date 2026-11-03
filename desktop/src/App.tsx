import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import type { AuditFinding, AuditReport, BoundingBox, ImageMeta, Page, SystemHealth, ToolType } from './types';
import {
  fetchHealth,
  fetchImages,
  fetchAllImages,
  fetchAnnotations,
  saveAnnotations,
  detectYolo,
  runAudit,
  imageIdentity,
  registerDatasetClass,
  closeProject,
  changeDatasetClass,
  type ClassChange,
  refreshColabAccount,
} from './services/api';
import { Sidebar } from './components/shell/Sidebar';
import { Topbar } from './components/shell/Topbar';
import { AnnotationCanvas } from './components/studio/AnnotationCanvas';
import { AnnotationPanel } from './components/studio/AnnotationPanel';
import { AuditPanel } from './components/studio/AuditPanel';
import { Filmstrip } from './components/studio/Filmstrip';
import { Badge, Button, Icon } from './components/ui';
import { applyFinding, applyOcclusionFlags, auditViews, nextEnhancePreset, visibleFindings } from './audit';
import { activatesFocusedControl, isTypingTarget, mergeDetections, statusAfterEdit, type ReviewStatus } from './review';
import { AutoLabelPage } from './pages/AutoLabelPage';
import { OverviewPage } from './pages/OverviewPage';
import { ProjectsPage } from './pages/ProjectsPage';
import { ImportPage } from './pages/ImportPage';
import { QualityPage } from './pages/QualityPage';
import { ExportPage } from './pages/ExportPage';
import { datasetNameFrom, summarizeDataset } from './pages/datasetSummary';
import { rememberDataset } from './pages/recentDatasets';
import { KeyboardShortcutsModal } from './components/modals/KeyboardShortcutsModal';
import { ColabModal } from './components/modals/ColabModal';

const PAGES: Page[] = ['overview', 'projects', 'import', 'autolabel', 'review', 'quality', 'export'];

const CRUMBS: Record<Page, string> = {
  overview: 'Ringkasan',
  projects: 'Proyek',
  import: 'Impor dataset',
  autolabel: 'Auto-label',
  review: 'Review anotasi',
  quality: 'Kualitas dataset',
  export: 'Ekspor dataset',
};

/** Start page from the current page-based URL. */
function initialPage(params: URLSearchParams | null): Page {
  const requested = params?.get('page') as Page | null;
  if (requested && PAGES.includes(requested)) return requested;
  return 'overview';
}

function statusBadge(image: ImageMeta | null) {
  if (!image) return null;
  if (image.status === 'reviewed') return <Badge tone="ok">Disetujui</Badge>;
  if (image.status === 'ai_labeled') return <Badge tone="warn">Auto-label · perlu review</Badge>;
  return <Badge tone="warn">Perlu review</Badge>;
}

export const App: React.FC = () => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [images, setImages] = useState<ImageMeta[]>([]);
  const [currentImageIndex, setCurrentImageIndex] = useState<number>(0);
  const [boxes, setBoxes] = useState<BoundingBox[]>([]);
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null);
  const [activeTool, setActiveTool] = useState<ToolType>('bbox');
  const [activeClassName, setActiveClassName] = useState<string>('object');

  const urlParams = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null;
  const [page, setPage] = useState<Page>(() => initialPage(urlParams));
  const [isShortcutsOpen, setIsShortcutsOpen] = useState(urlParams?.get('modal') === 'shortcuts');
  const [isColabOpen, setIsColabOpen] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [auditReport, setAuditReport] = useState<AuditReport | null>(null);
  const [enhance, setEnhance] = useState<string>('none');
  const [flipDetect, setFlipDetect] = useState(false);
  const flipDetectRef = useRef(flipDetect);
  flipDetectRef.current = flipDetect;
  const [auditDepth, setAuditDepth] = useState<string>('thorough');
  const auditDepthRef = useRef<string>(auditDepth);
  auditDepthRef.current = auditDepth;
  const [isAuditOpen, setIsAuditOpen] = useState(false);
  const [isAuditing, setIsAuditing] = useState(false);
  const [panelTab, setPanelTab] = useState<'annotations' | 'audit'>('annotations');
  const [selectedFindingId, setSelectedFindingId] = useState<string | null>(null);
  const [dismissedFindings, setDismissedFindings] = useState<ReadonlySet<string>>(new Set());
  const [hudToast, setHudToast] = useState<{ message: string; icon: string } | null>(null);
  const toastTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showToast = useCallback((message: string, icon: string = 'info') => {
    if (toastTimeoutRef.current) clearTimeout(toastTimeoutRef.current);
    setHudToast({ message, icon });
    toastTimeoutRef.current = setTimeout(() => {
      setHudToast(null);
    }, 1800);
  }, []);

  const pageRef = useRef<Page>(page);
  pageRef.current = page;
  const isColabOpenRef = useRef<boolean>(isColabOpen);
  isColabOpenRef.current = isColabOpen;

  const [loadedImageKey, setLoadedImageKey] = useState('');
  const [annotationReload, setAnnotationReload] = useState(0);
  const [saveError, setSaveError] = useState<{ image: ImageMeta; message: string } | null>(null);
  const draftsRef = useRef(new Map<string, BoundingBox[]>());
  const revisionsRef = useRef(new Map<string, number>());
  const writesRef = useRef(new Map<string, Promise<void>>());
  const pendingStatusRef = useRef(new Map<string, ReviewStatus>());
  const refreshGeneration = useRef(0);
  const currentImage = images[currentImageIndex] || null;
  const currentImageKey = currentImage ? imageIdentity(currentImage) : '';
  const loadedImageKeyRef = useRef(loadedImageKey);
  loadedImageKeyRef.current = loadedImageKey;
  const selectedBox = boxes.find((b) => b.id === selectedBoxId) || null;

  const healthRef = useRef<SystemHealth | null>(health);
  healthRef.current = health;
  const boxesRef = useRef<BoundingBox[]>(boxes);
  boxesRef.current = boxes;
  const selectedBoxIdRef = useRef<string | null>(selectedBoxId);
  selectedBoxIdRef.current = selectedBoxId;
  const selectedBoxRef = useRef<BoundingBox | null>(selectedBox);
  selectedBoxRef.current = selectedBox;
  const currentImageRef = useRef<ImageMeta | null>(currentImage);
  currentImageRef.current = currentImage;
  const imagesRef = useRef<ImageMeta[]>(images);
  imagesRef.current = images;
  const isDetectingRef = useRef<boolean>(isDetecting);
  isDetectingRef.current = isDetecting;
  const isShortcutsOpenRef = useRef<boolean>(isShortcutsOpen);
  isShortcutsOpenRef.current = isShortcutsOpen;
  const [classChangeBusy, setClassChangeBusy] = useState(false);
  const classChangeBusyRef = useRef(false);

  const refreshImages = useCallback(async () => {
    const generation = ++refreshGeneration.current;
    try {
      const h = await fetchHealth();
      const first = await fetchImages(200, 0);
      if (generation !== refreshGeneration.current || h.dataset_dir !== first.directory) return;
      setHealth(h);
      setActiveClassName(previous => h.classes.includes(previous) ? previous : h.classes[0] || '');
      setImages(first.images);
      setCurrentImageIndex(previous => Math.max(0, Math.min(previous, first.images.length - 1)));
      if (first.total > 200) {
        const all = await fetchAllImages();
        if (generation === refreshGeneration.current && all.directory === first.directory) setImages(all.images);
      }
    } catch (err) { showToast(`Dataset tidak bisa dimuat: ${String(err)}`, 'error'); }
  }, [showToast]);

  useEffect(() => { void refreshImages(); }, [refreshImages]);

  // Dropping a file anywhere except the import drop zone would make the webview navigate to it.
  useEffect(() => {
    const keepInApp = (event: DragEvent) => {
      if (event.dataTransfer?.types?.includes('Files')) event.preventDefault();
    };
    window.addEventListener('dragover', keepInApp);
    window.addEventListener('drop', keepInApp);
    return () => {
      window.removeEventListener('dragover', keepInApp);
      window.removeEventListener('drop', keepInApp);
    };
  }, []);
  useEffect(() => { if (health?.dataset_dir) rememberDataset(health.dataset_dir); }, [health?.dataset_dir]);

  useEffect(() => {
    let active = true;
    setBoxes([]);
    setSelectedBoxId(null);
    setLoadedImageKey('');
    setAuditReport(null);
    setSelectedFindingId(null);
    setDismissedFindings(new Set());
    if (!currentImage) return;
    const image = currentImage;
    const key = imageIdentity(image);
    const draft = draftsRef.current.get(key);
    if (draft) {
      setBoxes(draft);
      setLoadedImageKey(key);
      return;
    }
    fetchAnnotations(key, image.dataset_id).then(data => {
      if (!active) return;
      revisionsRef.current.set(key, data.revision);
      setBoxes(data.boxes || []);
      setSelectedBoxId(data.boxes?.[0]?.id || null);
      setLoadedImageKey(key);
    }).catch(err => { if (active) showToast(`Anotasi tidak bisa dimuat: ${String(err)}`, 'error'); });
    return () => { active = false; };
  }, [currentImageKey, currentImage?.dataset_id, annotationReload, showToast]);

  const persistBoxes = useCallback(async (updatedBoxes: BoundingBox[], target?: ImageMeta, status?: ReviewStatus) => {
    const image = target || currentImageRef.current;
    if (!image) return;
    const key = imageIdentity(image);
    if (!target && loadedImageKeyRef.current !== key) return;
    // Editing keeps the image's review state; only an explicit approval changes it. Writes for
    // one image are queued, so a pending approval must not be undone by an edit made right after.
    const known = imagesRef.current.find(item => imageIdentity(item) === key) || image;
    const saveStatus = status ?? pendingStatusRef.current.get(key) ?? statusAfterEdit(known.status);
    pendingStatusRef.current.set(key, saveStatus);
    draftsRef.current.set(key, updatedBoxes);
    if (currentImageRef.current && imageIdentity(currentImageRef.current) === key) {
      boxesRef.current = updatedBoxes;
      setBoxes(updatedBoxes);
    }
    const previous = writesRef.current.get(key) || Promise.resolve();
    const write = previous.then(async () => {
      try {
        const result = await saveAnnotations(key, updatedBoxes, revisionsRef.current.get(key), image.dataset_id, saveStatus);
        revisionsRef.current.set(key, result.revision);
        if (result.projection_warning) showToast(result.projection_warning, 'error');
        if (draftsRef.current.get(key) === updatedBoxes) draftsRef.current.delete(key);
        setImages(prev => prev.map(img => imageIdentity(img) === key
          ? { ...img, annotation_count: updatedBoxes.length, status: result.review_status ?? saveStatus } : img));
        setSaveError(previousError => previousError && imageIdentity(previousError.image) === key ? null : previousError);
      } catch (err) {
        setSaveError({ image, message: `Perubahan pada ${image.filename} belum tersimpan: ${String(err)}` });
      }
    });
    writesRef.current.set(key, write);
    await write;
    if (writesRef.current.get(key) === write) {
      writesRef.current.delete(key);
      pendingStatusRef.current.delete(key);
    }
  }, [showToast]);

  useEffect(() => {
    const warnUnsaved = (event: BeforeUnloadEvent) => {
      if (draftsRef.current.size) { event.preventDefault(); event.returnValue = ''; }
    };
    window.addEventListener('beforeunload', warnUnsaved);
    return () => window.removeEventListener('beforeunload', warnUnsaved);
  }, []);

  const handleAddBox = (newBox: BoundingBox) => {
    const updated = [...boxesRef.current, newBox];
    persistBoxes(updated);
  };

  const handleUpdateBox = (updatedBox: BoundingBox) => {
    const updated = boxesRef.current.map((b) => (b.id === updatedBox.id ? updatedBox : b));
    persistBoxes(updated);
  };

  const handleDeleteBox = (id: string) => {
    const updated = boxesRef.current.filter((b) => b.id !== id);
    if (selectedBoxIdRef.current === id) {
      setSelectedBoxId(updated[0]?.id || null);
    }
    persistBoxes(updated);
  };

  const handleAcceptBox = (id: string) => {
    const box = boxesRef.current.find((b) => b.id === id);
    if (box) {
      handleUpdateBox({ ...box, confidence: 1.0 });
    }
  };

  const handleRunYoloDetect = async () => {
    const activeImg = currentImageRef.current;
    if (!activeImg || loadedImageKeyRef.current !== imageIdentity(activeImg) || isDetectingRef.current) return;
    const startingBoxes = boxesRef.current;
    setIsDetecting(true);
    try {
      const res = await detectYolo(
        imageIdentity(activeImg), 0.25, undefined, activeImg.dataset_id,
        flipDetectRef.current && healthRef.current?.inference?.backend !== 'cloud' ? ['original', 'flip'] : undefined,
      );
      if (!currentImageRef.current || imageIdentity(currentImageRef.current) !== imageIdentity(activeImg)) {
        showToast('Deteksi selesai untuk gambar lain. Jalankan lagi di gambar itu untuk menerapkannya.', 'info');
        return;
      }
      if (boxesRef.current !== startingBoxes) {
        showToast('Anotasi berubah selama deteksi. Jalankan deteksi lagi untuk menerapkannya.', 'info');
        return;
      }
      // Keep every existing label: a detector pass only adds objects that are not labelled yet.
      const merged = mergeDetections(startingBoxes, res.boxes);
      if (merged.added > 0) {
        await persistBoxes(merged.boxes, activeImg);
        setSelectedBoxId(merged.boxes[startingBoxes.length]?.id || null);
      }
      showToast(
        merged.added > 0
          ? `${merged.added} kotak baru dari deteksi AI${merged.skipped ? ` (${merged.skipped} sudah dilabeli)` : ''}`
          : 'Tidak ada objek baru yang terdeteksi',
        merged.added > 0 ? 'auto_awesome' : 'info',
      );
      const h = await fetchHealth();
      if (h.dataset_dir === currentImageRef.current?.dataset_id) setHealth(h);
    } catch (err) {
      showToast(`Deteksi gagal: ${String(err)}`, 'error');
    } finally {
      setIsDetecting(false);
    }
  };

  const findings = visibleFindings(auditReport, boxes, dismissedFindings);
  const findingsRef = useRef<AuditFinding[]>(findings);
  findingsRef.current = findings;
  const selectedFindingIdRef = useRef<string | null>(selectedFindingId);
  selectedFindingIdRef.current = selectedFindingId;
  const isAuditOpenRef = useRef<boolean>(isAuditOpen);
  isAuditOpenRef.current = isAuditOpen;
  const isAuditingRef = useRef<boolean>(isAuditing);
  isAuditingRef.current = isAuditing;
  const auditReportRef = useRef<AuditReport | null>(auditReport);
  auditReportRef.current = auditReport;

  const handleRunAudit = async () => {
    const image = currentImageRef.current;
    if (!image || loadedImageKeyRef.current !== imageIdentity(image) || isAuditingRef.current) return;
    const key = imageIdentity(image);
    const startingBoxes = boxesRef.current;
    setIsAuditOpen(true);
    setPanelTab('audit');
    setIsAuditing(true);
    try {
      const report = await runAudit(key, startingBoxes, image.dataset_id, auditViews(auditDepthRef.current));
      if (!currentImageRef.current || imageIdentity(currentImageRef.current) !== key) return;
      if (boxesRef.current !== startingBoxes) {
        showToast('Anotasi berubah selama audit. Jalankan lagi.', 'info');
        return;
      }
      setAuditReport(report);
      setDismissedFindings(new Set());
      setSelectedFindingId(report.findings[0]?.id || null);
    } catch (err) {
      showToast(`Audit gagal: ${String(err)}`, 'error');
    } finally {
      setIsAuditing(false);
    }
  };

  const selectFinding = (id: string | null) => {
    setSelectedFindingId(id);
    const finding = findingsRef.current.find(item => item.id === id);
    if (finding?.annotation_id) setSelectedBoxId(finding.annotation_id);
  };

  const moveFindingSelection = (step: number) => {
    const list = findingsRef.current;
    if (!list.length) return;
    const index = list.findIndex(item => item.id === selectedFindingIdRef.current);
    selectFinding(list[(index + step + list.length) % list.length].id);
  };

  const dismissFinding = (finding: AuditFinding) => {
    const list = findingsRef.current;
    const index = list.findIndex(item => item.id === finding.id);
    setDismissedFindings(previous => new Set(previous).add(finding.id));
    selectFinding(list[index + 1]?.id ?? list[index - 1]?.id ?? null);
  };

  const applyAuditFinding = (finding: AuditFinding) => {
    const updated = applyFinding(finding, boxesRef.current, healthRef.current?.classes || []);
    if (!updated) {
      showToast('Saran ini tidak bisa diterapkan (kelasnya belum ada di dataset?)', 'error');
      return;
    }
    if (finding.annotation_id && !updated.some(box => box.id === finding.annotation_id)
        && selectedBoxIdRef.current === finding.annotation_id) setSelectedBoxId(null);
    // Applying a fix changes the boxes, so the remaining findings for that box are stale.
    setDismissedFindings(previous => {
      const next = new Set(previous).add(finding.id);
      findingsRef.current.forEach(item => {
        if (finding.annotation_id && item.annotation_id === finding.annotation_id) next.add(item.id);
      });
      return next;
    });
    const list = findingsRef.current;
    const index = list.findIndex(item => item.id === finding.id);
    const unrelated = (item: AuditFinding) => !finding.annotation_id || item.annotation_id !== finding.annotation_id;
    const next = [...list.slice(index + 1), ...list.slice(0, index)].find(unrelated);
    setSelectedFindingId(next?.id ?? null);
    void persistBoxes(updated);
  };

  const flagOccludedBoxes = () => {
    const updated = applyOcclusionFlags(boxesRef.current, auditReportRef.current);
    if (!updated) return;
    void persistBoxes(updated);
    showToast('Kotak tertutup sudah ditandai', 'visibility_off');
  };

  const unflaggedOcclusionCount = auditReport
    ? auditReport.occlusion.filter(item => item.occluded
        && boxes.some(box => box.id === item.annotation_id && !box.occluded)).length
    : 0;

  const openImage = (index: number) => {
    setCurrentImageIndex(Math.max(0, Math.min(index, imagesRef.current.length - 1)));
    setPage('review');
  };

  const goToImage = (step: number) => {
    setCurrentImageIndex(previous => Math.max(0, Math.min(imagesRef.current.length - 1, previous + step)));
  };

  /** Save the current boxes, which marks the image as approved, and optionally move on. */
  const approveImage = async (advance: boolean) => {
    const image = currentImageRef.current;
    if (!image || loadedImageKeyRef.current !== imageIdentity(image)) return;
    await persistBoxes(boxesRef.current, undefined, 'reviewed');
    showToast(`${image.filename} disetujui`, 'check_circle');
    if (!advance) return;
    const list = imagesRef.current;
    const from = list.findIndex(item => imageIdentity(item) === imageIdentity(image));
    const next = list.findIndex((item, index) => index > from && item.status !== 'reviewed');
    if (next >= 0) setCurrentImageIndex(next);
    else if (from < list.length - 1) setCurrentImageIndex(from + 1);
  };

  const closeAudit = () => {
    setIsAuditOpen(false);
    setPanelTab('annotations');
  };

  const datasetChanged = () => {
    setImages([]); setCurrentImageIndex(0); setBoxes([]); setLoadedImageKey('');
    void refreshImages();
  };

  const manageClasses = async (change: ClassChange) => {
    if (classChangeBusyRef.current || isDetectingRef.current || isAuditingRef.current) {
      throw new Error('Tunggu deteksi, audit, atau perubahan kelas selesai.');
    }
    classChangeBusyRef.current = true;
    setClassChangeBusy(true);
    try {
      await Promise.all([...writesRef.current.values()]);
      if (draftsRef.current.size) {
        throw new Error('Masih ada anotasi yang belum tersimpan. Simpan ulang sebelum mengubah kelas.');
      }
      const result = await changeDatasetClass(change);
      if (healthRef.current?.dataset_dir === change.dataset_id) {
        setHealth(previous => previous ? { ...previous, classes: result.classes } : previous);
        setActiveClassName(previous => result.classes.includes(previous) ? previous : result.classes[0] || '');
        setBoxes([]);
        setLoadedImageKey('');
        setSelectedBoxId(null);
        setAuditReport(null);
        setAnnotationReload(value => value + 1);
      }
      await refreshImages();
      return result;
    } finally {
      classChangeBusyRef.current = false;
      setClassChangeBusy(false);
    }
  };

  const closeActiveProject = async (folder: string) => {
    if (isDetectingRef.current || isAuditingRef.current) {
      throw new Error('Tunggu deteksi atau audit selesai sebelum menghapus proyek.');
    }
    await Promise.all([...writesRef.current.values()]);
    if (draftsRef.current.size) {
      throw new Error('Masih ada anotasi yang belum tersimpan. Simpan ulang sebelum menghapus proyek.');
    }
    await closeProject(folder);
    ++refreshGeneration.current;
    setHealth(previous => previous ? { ...previous, dataset_dir: '', classes: [] } : previous);
    setSelectedBoxId(null);
    setAuditReport(null);
    setIsAuditOpen(false);
    datasetChanged();
  };

  const openColab = () => {
    setIsColabOpen(true);
    void refreshColabAccount().catch(() => { /* The dialog reports connection errors. */ });
  };

  // Keyboard shortcuts. Studio keys only act on the review page so nothing changes out of sight.
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (classChangeBusyRef.current) return;
      // Typing, choosing an option or dragging a slider is not a shortcut, but ticking a
      // checkbox must not turn the shortcuts off. Enter/Space on a focused control activates it.
      if (isTypingTarget(e.target) || activatesFocusedControl(e.target, e.key)) {
        return;
      }

      if (e.key === 'Escape') {
        if (isShortcutsOpenRef.current) setIsShortcutsOpen(false);
        else if (isColabOpenRef.current) setIsColabOpen(false);
        else if (pageRef.current === 'review' && selectedBoxIdRef.current) {
          setSelectedBoxId(null);
          showToast('Pilihan kotak dibatalkan', 'close');
        }
        return;
      }
      if (e.key === '?' || e.key === 'F1') {
        setIsShortcutsOpen((prev) => !prev);
        return;
      }
      if (isShortcutsOpenRef.current || isColabOpenRef.current) return;
      if ((e.ctrlKey || e.metaKey) && !['s', 'S', 'Enter'].includes(e.key)) return;

      // Page navigation: Auto-label (L / M), Import (I), Export (E)
      if (!e.ctrlKey && !e.metaKey && !e.altKey) {
        const target: Page | null = ['l', 'L', 'm', 'M'].includes(e.key) ? 'autolabel'
          : e.key === 'i' || e.key === 'I' ? 'import'
          : e.key === 'e' || e.key === 'E' ? 'export' : null;
        if (target) {
          setPage(target);
          return;
        }
      }
      if (pageRef.current !== 'review') return;

      const curBoxes = boxesRef.current;
      const curSelectedId = selectedBoxIdRef.current;
      const curSelectedBox = selectedBoxRef.current;
      const curImages = imagesRef.current;

      // 1. Tool Selection
      if (e.key === 'v' || e.key === 'V') {
        setActiveTool('select');
        showToast('Pilih & ubah (V)', 'near_me');
      } else if (e.key === 'b' || e.key === 'B') {
        setActiveTool('bbox');
        showToast('Kotak (B)', 'crop_free');
      } else if (e.key === 'h' || e.key === 'H') {
        setActiveTool('pan');
        showToast('Geser kanvas (H)', 'pan_tool');
      } else if (e.key === 'p' || e.key === 'P') {
        setActiveTool('polygon');
        showToast('Poligon (P)', 'polyline');
      }

      // 2. Image Navigation
      else if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft' || e.key === '[') {
        setCurrentImageIndex((prev) => {
          const next = Math.max(0, prev - 1);
          if (next !== prev) showToast(`Gambar ${next + 1} dari ${curImages.length}`, 'arrow_back');
          return next;
        });
      } else if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight' || e.key === ']') {
        setCurrentImageIndex((prev) => {
          const next = Math.min(curImages.length - 1, prev + 1);
          if (next !== prev) showToast(`Gambar ${next + 1} dari ${curImages.length}`, 'arrow_forward');
          return next;
        });
      } else if (e.key === 'Home') {
        setCurrentImageIndex(0);
      } else if (e.key === 'End') {
        setCurrentImageIndex(curImages.length - 1);
      }

      // 3. Selection Cycling (Tab / Shift+Tab)
      else if (e.key === 'Tab') {
        e.preventDefault();
        if (curBoxes.length === 0) return;
        const currentIndex = curBoxes.findIndex((b) => b.id === curSelectedId);
        const nextIdx = e.shiftKey
          ? (currentIndex <= 0 ? curBoxes.length - 1 : currentIndex - 1)
          : (currentIndex === -1 || currentIndex >= curBoxes.length - 1 ? 0 : currentIndex + 1);
        setSelectedBoxId(curBoxes[nextIdx].id);
        showToast(`Anotasi ${nextIdx + 1}: ${curBoxes[nextIdx].class_name}`, 'tab');
      }

      // 4. Box Deletion (Delete / Backspace)
      else if ((e.key === 'Delete' || e.key === 'Backspace') && curSelectedId) {
        e.preventDefault();
        handleDeleteBox(curSelectedId);
        showToast('Kotak dihapus', 'delete');
      }

      // 5. Box Acceptance / Triage (Enter / Space)
      else if ((e.key === 'Enter' || e.key === ' ') && curSelectedId && !e.ctrlKey) {
        e.preventDefault();
        handleAcceptBox(curSelectedId);
        showToast('Kotak ditandai benar', 'check_circle');
      }

      // 6. Quick Class Assignment (Keys 1-9)
      else if (/^[1-9]$/.test(e.key)) {
        const availableClasses = healthRef.current?.classes || [];
        const keyIndex = parseInt(e.key, 10) - 1;
        if (keyIndex < availableClasses.length) {
          const targetName = availableClasses[keyIndex];
          setActiveClassName(targetName);
          if (curSelectedBox) {
            handleUpdateBox({ ...curSelectedBox, class_name: targetName, class_id: keyIndex });
          }
          showToast(`Kelas: ${targetName}`, 'label');
        }
      }

      // 7. AI Detection (R / Ctrl+Enter)
      else if (e.key === 'r' || e.key === 'R' || (e.ctrlKey && e.key === 'Enter')) {
        if (!isDetectingRef.current) {
          showToast('Menjalankan deteksi AI… (R)', 'auto_awesome');
          handleRunYoloDetect();
        }
      }

      // 8. Save / approve (Ctrl+S / Cmd+S) through the same queue as every other edit
      else if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S')) {
        e.preventDefault();
        void approveImage(false);
      }

      // 8b. Annotation audit: Q run, J/K step through findings, Y apply, N dismiss
      else if (e.key === 'q' || e.key === 'Q') {
        void handleRunAudit();
      } else if (isAuditOpenRef.current && (e.key === 'j' || e.key === 'J')) {
        moveFindingSelection(1);
      } else if (isAuditOpenRef.current && (e.key === 'k' || e.key === 'K')) {
        moveFindingSelection(-1);
      } else if (isAuditOpenRef.current && (e.key === 'y' || e.key === 'Y' || e.key === 'n' || e.key === 'N')) {
        const finding = findingsRef.current.find(item => item.id === selectedFindingIdRef.current);
        if (finding) {
          if (e.key === 'y' || e.key === 'Y') applyAuditFinding(finding);
          else dismissFinding(finding);
        }
      }

      // 8c. Cycle the display-only image filter
      else if (e.key === 'f' || e.key === 'F') {
        setEnhance((previous) => {
          const next = nextEnhancePreset(previous);
          showToast(`Filter gambar: ${next}`, 'deblur');
          return next;
        });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showToast]);

  const summary = useMemo(() => summarizeDataset(images), [images]);
  const datasetName = datasetNameFrom(health?.dataset_dir);
  const classes = health?.classes || [];
  const refreshAfterLabels = () => { void refreshImages(); setAnnotationReload(value => value + 1); };

  const review = (
    <div className="mx-auto flex min-h-full max-w-[1400px] flex-col px-6 py-6 xl:h-full">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="text-xs text-ink-500">{datasetName || 'Dataset'}</div>
          <h1 className="mt-1 truncate text-2xl font-bold tracking-tight text-ink-900" title={currentImage?.filename}>
            {currentImage?.filename || 'Belum ada gambar'}
          </h1>
          <div className="mt-1.5 text-xs text-ink-500">
            Gambar {images.length ? currentImageIndex + 1 : 0} dari {images.length}
            <span className="mx-2 text-ink-300">/</span>
            {boxes.length} anotasi
          </div>
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-3">
          {statusBadge(currentImage)}
          <Button variant="secondary" icon="rule" disabled={!currentImage || isAuditing} onClick={() => void handleRunAudit()}
            title="Cari objek terlewat, tertutup, dan kotak yang salah (Q)">
            {isAuditing ? 'Memeriksa…' : 'Cek kualitas'}
          </Button>
          <Button disabled={!currentImage || loadedImageKey !== currentImageKey} onClick={() => void approveImage(true)}
            title="Setujui gambar ini lalu buka gambar berikutnya yang perlu review">
            Simpan &amp; lanjut
          </Button>
        </div>
      </div>

      <div className="flex min-h-0 flex-1 flex-col gap-5 xl:flex-row">
        <AnnotationCanvas
          image={currentImage}
          boxes={boxes}
          selectedBoxId={selectedBoxId}
          activeTool={activeTool}
          activeClassName={activeClassName}
          availableClasses={classes}
          readOnly={loadedImageKey !== currentImageKey}
          onSelectBox={setSelectedBoxId}
          onSelectTool={setActiveTool}
          onAddBox={handleAddBox}
          onUpdateBox={handleUpdateBox}
          onRunYoloDetect={handleRunYoloDetect}
          isDetecting={isDetecting}
          auditFindings={isAuditOpen ? findings : []}
          selectedFindingId={selectedFindingId}
          isAuditing={isAuditing}
          onSelectFinding={selectFinding}
          onRunAudit={() => void handleRunAudit()}
          enhance={enhance}
          onChangeEnhance={setEnhance}
          flipDetect={flipDetect && health?.inference?.backend !== 'cloud'}
          flipDetectDisabled={health?.inference?.backend === 'cloud'}
          onChangeFlipDetect={setFlipDetect}
          footer={<Filmstrip images={images} currentIndex={currentImageIndex} onSelect={setCurrentImageIndex} />}
        />

        <AnnotationPanel
          boxes={boxes}
          selectedBoxId={selectedBoxId}
          availableClasses={classes}
          activeClassName={activeClassName}
          onSelectBox={setSelectedBoxId}
          onUpdateBox={handleUpdateBox}
          onDeleteBox={handleDeleteBox}
          onAcceptBox={handleAcceptBox}
          onChangeActiveClass={setActiveClassName}
          tab={isAuditOpen ? panelTab : 'annotations'}
          onTabChange={setPanelTab}
          onRegisterClass={async name => {
            const image = currentImageRef.current;
            const selected = selectedBoxRef.current;
            if (!image || !selected) throw new Error('Pilih anotasi dulu');
            const registered = await registerDatasetClass(name, image.dataset_id);
            if (imageIdentity(currentImageRef.current || image) !== imageIdentity(image)
                || selectedBoxRef.current !== selected) throw new Error('Pilihan berubah; pakai kelasnya lagi');
            setHealth(previous => previous && previous.dataset_dir === image.dataset_id ? { ...previous, classes: registered } : previous);
            return registered.findIndex(c => c.toLowerCase() === name.toLowerCase());
          }}
          audit={isAuditOpen ? {
            count: findings.length,
            content: (
              <AuditPanel
                report={auditReport}
                findings={findings}
                selectedFindingId={selectedFindingId}
                isRunning={isAuditing}
                unflaggedOcclusionCount={unflaggedOcclusionCount}
                depth={auditDepth}
                onChangeDepth={setAuditDepth}
                onRun={() => void handleRunAudit()}
                onClose={closeAudit}
                onSelect={selectFinding}
                onApply={applyAuditFinding}
                onDismiss={dismissFinding}
                onFlagOccluded={flagOccludedBoxes}
              />
            ),
          } : null}
        />
      </div>

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex gap-2">
          <Button variant="secondary" disabled={currentImageIndex <= 0} onClick={() => goToImage(-1)}>Sebelumnya</Button>
          <Button variant="secondary" disabled={currentImageIndex >= images.length - 1} onClick={() => goToImage(1)}>Berikutnya</Button>
        </div>
        <span className="text-xs text-ink-500">
          <b className="text-ink-900 tabular-nums">{summary.approved.toLocaleString('id-ID')}</b>/{summary.total.toLocaleString('id-ID')} gambar disetujui
        </span>
        <Button icon="check" disabled={!currentImage || loadedImageKey !== currentImageKey} onClick={() => void approveImage(false)}>
          Setujui gambar
        </Button>
      </div>
    </div>
  );

  return (
    <>
    <div inert={classChangeBusy} className="relative flex h-screen w-screen overflow-hidden bg-white font-sans text-ink-900">
      <Sidebar page={page} onNavigate={setPage} health={health} datasetName={datasetName} reviewCount={summary.pending} />

      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar crumb={CRUMBS[page]} health={health} onOpenShortcuts={() => setIsShortcutsOpen(true)} onOpenColab={openColab} />
        <main className="min-h-0 flex-1 overflow-y-auto">
          {page === 'overview' && (
            <OverviewPage images={images} health={health} datasetName={datasetName} onNavigate={setPage} onOpenImage={openImage} />
          )}
          {page === 'projects' && (
            <ProjectsPage images={images} health={health} datasetName={datasetName} onNavigate={setPage}
              onDatasetChanged={datasetChanged} onCloseProject={closeActiveProject} />
          )}
          {page === 'import' && (
            <ImportPage images={images} health={health} datasetName={datasetName} onNavigate={setPage}
              onDatasetChanged={datasetChanged}
              onClassesChanged={registered => setHealth(previous => previous ? { ...previous, classes: registered } : previous)}
              onManageClasses={manageClasses}
              onAnnotationsImported={refreshAfterLabels} />
          )}
          {page === 'review' && review}
          {page === 'quality' && <QualityPage images={images} onNavigate={setPage} onOpenImage={openImage} />}
          {page === 'export' && <ExportPage images={images} health={health} datasetName={datasetName} onNavigate={setPage} />}
          {/* Stays mounted so a running batch keeps reporting progress while you browse other pages. */}
          <AutoLabelPage
            isOpen={page === 'autolabel'}
            onClose={() => undefined}
            onNavigate={setPage}
            onOpenColab={openColab}
            health={health}
            totalImages={images.length}
            pendingImages={summary.unannotated}
            images={images}
            onStartBatch={refreshAfterLabels}
            onBatchComplete={refreshAfterLabels}
          />
        </main>
      </div>

      {saveError && (
        <div role="alert" className="fixed left-1/2 top-16 z-50 flex max-w-xl -translate-x-1/2 items-center gap-3 rounded-lg border border-danger-600/30 bg-danger-50 px-4 py-3 text-sm text-danger-600 shadow-lg">
          <Icon name="error" className="text-[18px]" />
          <span className="min-w-0 flex-1">{saveError.message}</span>
          <button className="font-semibold underline cursor-pointer" onClick={() => {
            const draft = draftsRef.current.get(imageIdentity(saveError.image));
            if (draft) void persistBoxes(draft, saveError.image);
          }}>Simpan ulang</button>
        </div>
      )}
      {hudToast && (
        <div className="pointer-events-none fixed bottom-6 left-1/2 z-50 -translate-x-1/2">
          <div className="flex items-center gap-2 rounded-full bg-ink-900 px-4 py-2 text-xs font-medium text-white shadow-xl">
            <Icon name={hudToast.icon} className="text-[16px] text-brand-300" />
            {hudToast.message}
          </div>
        </div>
      )}

      <ColabModal isOpen={isColabOpen} onClose={() => setIsColabOpen(false)} health={health} onComplete={refreshAfterLabels} />
      <KeyboardShortcutsModal isOpen={isShortcutsOpen} onClose={() => setIsShortcutsOpen(false)} />
    </div>
    {classChangeBusy && <div role="status" aria-live="polite" className="fixed inset-0 z-50 flex items-center justify-center bg-white/95 p-6 text-sm font-medium text-ink-900">
      Memperbarui kelas dan anotasi… Jangan tutup aplikasi.
    </div>}
    </>
  );
};

export default App;
