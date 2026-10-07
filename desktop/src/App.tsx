import React, { useState, useEffect, useCallback, useRef } from 'react';
import type { BoundingBox, ImageMeta, SystemHealth, ToolType } from './types';
import {
  fetchHealth,
  fetchImages,
  fetchAllImages,
  fetchAnnotations,
  saveAnnotations,
  detectYolo,
  imageIdentity,
  registerDatasetClass,
} from './services/api';
import { StudioHeader } from './components/studio/StudioHeader';
import { ActivityRail } from './components/studio/ActivityRail';
import { ImageQueueDrawer } from './components/studio/ImageQueueDrawer';
import { AnnotationCanvas } from './components/studio/AnnotationCanvas';
import { ObjectInspector } from './components/studio/ObjectInspector';
import { AutoLabelModal } from './components/modals/AutoLabelModal';
import { DatasetHubModal } from './components/modals/DatasetHubModal';
import { KeyboardShortcutsModal } from './components/modals/KeyboardShortcutsModal';

export const App: React.FC = () => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [images, setImages] = useState<ImageMeta[]>([]);
  const [currentImageIndex, setCurrentImageIndex] = useState<number>(0);
  const [boxes, setBoxes] = useState<BoundingBox[]>([]);
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null);
  const [activeTool, setActiveTool] = useState<ToolType>('bbox');
  const [activeClassName, setActiveClassName] = useState<string>('object');

  const urlParams = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null;
  const [isAutoLabelOpen, setIsAutoLabelOpen] = useState(urlParams?.get('modal') === 'autolabel');
  const [isDatasetHubOpen, setIsDatasetHubOpen] = useState(urlParams?.get('modal') === 'dataset');
  const [isShortcutsOpen, setIsShortcutsOpen] = useState(urlParams?.get('modal') === 'shortcuts');
  const [datasetHubTab, setDatasetHubTab] = useState<'export' | 'import'>(
    urlParams?.get('tab') === 'export' ? 'export' : 'import'
  );
  const [isDetecting, setIsDetecting] = useState(false);
  const [hudToast, setHudToast] = useState<{ message: string; icon: string } | null>(null);
  const toastTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showToast = useCallback((message: string, icon: string = 'info') => {
    if (toastTimeoutRef.current) clearTimeout(toastTimeoutRef.current);
    setHudToast({ message, icon });
    toastTimeoutRef.current = setTimeout(() => {
      setHudToast(null);
    }, 1800);
  }, []);

  const [loadedImageKey, setLoadedImageKey] = useState('');
  const [annotationReload, setAnnotationReload] = useState(0);
  const [saveError, setSaveError] = useState<{ image: ImageMeta; message: string } | null>(null);
  const draftsRef = useRef(new Map<string, BoundingBox[]>());
  const revisionsRef = useRef(new Map<string, number>());
  const writesRef = useRef(new Map<string, Promise<void>>());
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
  const isAutoLabelOpenRef = useRef<boolean>(isAutoLabelOpen);
  isAutoLabelOpenRef.current = isAutoLabelOpen;
  const isDatasetHubOpenRef = useRef<boolean>(isDatasetHubOpen);
  isDatasetHubOpenRef.current = isDatasetHubOpen;
  const isShortcutsOpenRef = useRef<boolean>(isShortcutsOpen);
  isShortcutsOpenRef.current = isShortcutsOpen;

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
    } catch (err) { showToast(`Cannot load dataset: ${String(err)}`, 'error'); }
  }, [showToast]);

  useEffect(() => { void refreshImages(); }, [refreshImages]);

  useEffect(() => {
    let active = true;
    setBoxes([]);
    setSelectedBoxId(null);
    setLoadedImageKey('');
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
    }).catch(err => { if (active) showToast(`Cannot load annotations: ${String(err)}`, 'error'); });
    return () => { active = false; };
  }, [currentImageKey, currentImage?.dataset_id, annotationReload, showToast]);

  const persistBoxes = useCallback(async (updatedBoxes: BoundingBox[], target?: ImageMeta) => {
    const image = target || currentImageRef.current;
    if (!image) return;
    const key = imageIdentity(image);
    if (!target && loadedImageKeyRef.current !== key) return;
    draftsRef.current.set(key, updatedBoxes);
    if (currentImageRef.current && imageIdentity(currentImageRef.current) === key) {
      boxesRef.current = updatedBoxes;
      setBoxes(updatedBoxes);
    }
    const previous = writesRef.current.get(key) || Promise.resolve();
    const write = previous.then(async () => {
      try {
        const result = await saveAnnotations(key, updatedBoxes, revisionsRef.current.get(key), image.dataset_id);
        revisionsRef.current.set(key, result.revision);
        if (result.projection_warning) showToast(result.projection_warning, 'error');
        if (draftsRef.current.get(key) === updatedBoxes) draftsRef.current.delete(key);
        setImages(prev => prev.map(img => imageIdentity(img) === key
          ? { ...img, annotation_count: updatedBoxes.length, status: 'reviewed' } : img));
        setSaveError(previousError => previousError && imageIdentity(previousError.image) === key ? null : previousError);
      } catch (err) {
        setSaveError({ image, message: `Edits for ${image.filename} are unsaved: ${String(err)}` });
      }
    });
    writesRef.current.set(key, write);
    await write;
    if (writesRef.current.get(key) === write) writesRef.current.delete(key);
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
      const res = await detectYolo(imageIdentity(activeImg), 0.25, undefined, activeImg.dataset_id);
      if (!currentImageRef.current || imageIdentity(currentImageRef.current) !== imageIdentity(activeImg)) {
        showToast('Detection finished for a different image. Run it again on that image to apply.', 'info');
        return;
      }
      if (boxesRef.current !== startingBoxes) {
        showToast('Annotations changed during detection. Run detection again to apply.', 'info');
        return;
      }
      await persistBoxes(res.boxes, activeImg);
      setSelectedBoxId(res.boxes[0]?.id || null);
      const h = await fetchHealth();
      if (h.dataset_dir === currentImageRef.current?.dataset_id) setHealth(h);
    } catch (err) {
      showToast(`Detection failed: ${String(err)}`, 'error');
    } finally {
      setIsDetecting(false);
    }
  };

  // Keyboard navigation and velocity shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in an input
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)) {
        return;
      }

      const curBoxes = boxesRef.current;
      const curSelectedId = selectedBoxIdRef.current;
      const curSelectedBox = selectedBoxRef.current;
      const curImages = imagesRef.current;
      const curImg = currentImageRef.current;

      // 1. Tool Selection
      if (e.key === 'v' || e.key === 'V') {
        setActiveTool('select');
        showToast('Select & Transform Tool (V)', 'near_me');
      } else if (e.key === 'b' || e.key === 'B') {
        setActiveTool('bbox');
        showToast('2D Bounding Box Tool (B)', 'crop_square');
      } else if (e.key === 'h' || e.key === 'H') {
        setActiveTool('pan');
        showToast('Pan Canvas Tool (H)', 'pan_tool');
      } else if (e.key === 'p' || e.key === 'P') {
        setActiveTool('polygon');
        showToast('Polygon Segmentation Tool (P)', 'polyline');
      }

      // 2. Image Navigation
      else if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft' || e.key === '[') {
        setCurrentImageIndex((prev) => {
          const next = Math.max(0, prev - 1);
          if (next !== prev) showToast(`Image ${next + 1} of ${curImages.length}`, 'arrow_back');
          return next;
        });
      } else if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight' || e.key === ']') {
        setCurrentImageIndex((prev) => {
          const next = Math.min(curImages.length - 1, prev + 1);
          if (next !== prev) showToast(`Image ${next + 1} of ${curImages.length}`, 'arrow_forward');
          return next;
        });
      } else if (e.key === 'Home') {
        setCurrentImageIndex(0);
        showToast(`First Image (1 of ${curImages.length})`, 'first_page');
      } else if (e.key === 'End') {
        setCurrentImageIndex(curImages.length - 1);
        showToast(`Last Image (${curImages.length} of ${curImages.length})`, 'last_page');
      }

      // 3. Selection Cycling (Tab / Shift+Tab)
      else if (e.key === 'Tab') {
        e.preventDefault();
        if (curBoxes.length === 0) return;
        const currentIndex = curBoxes.findIndex((b) => b.id === curSelectedId);
        if (e.shiftKey) {
          const prevIdx = currentIndex <= 0 ? curBoxes.length - 1 : currentIndex - 1;
          setSelectedBoxId(curBoxes[prevIdx].id);
          showToast(`Selected #${curBoxes[prevIdx].id.slice(-4)} (${curBoxes[prevIdx].class_name})`, 'tab');
        } else {
          const nextIdx = currentIndex === -1 || currentIndex >= curBoxes.length - 1 ? 0 : currentIndex + 1;
          setSelectedBoxId(curBoxes[nextIdx].id);
          showToast(`Selected #${curBoxes[nextIdx].id.slice(-4)} (${curBoxes[nextIdx].class_name})`, 'tab');
        }
      }

      // 4. Box Deletion (Delete / Backspace)
      else if ((e.key === 'Delete' || e.key === 'Backspace') && curSelectedId) {
        e.preventDefault();
        const idToDelete = curSelectedId;
        handleDeleteBox(idToDelete);
        showToast(`Deleted Box #${idToDelete.slice(-4)}`, 'delete');
      }

      // 5. Box Acceptance / Triage (Enter / Space)
      else if ((e.key === 'Enter' || e.key === ' ') && curSelectedId) {
        e.preventDefault();
        handleAcceptBox(curSelectedId);
        showToast(`Validated Box #${curSelectedId.slice(-4)} (100% Conf)`, 'check_circle');
      }

      // 6. Quick Class Assignment (Keys 1-6)
      else if (['1', '2', '3', '4', '5', '6'].includes(e.key)) {
        const availableClasses = healthRef.current?.classes && healthRef.current.classes.length > 0
          ? healthRef.current.classes
          : ['object', 'person', 'car', 'bus', 'truck', 'motorcycle'];
        const keyIndex = parseInt(e.key, 10) - 1;
        if (keyIndex >= 0 && keyIndex < availableClasses.length) {
          const targetName = availableClasses[keyIndex];
          setActiveClassName(targetName);
          if (curSelectedBox) {
            handleUpdateBox({
              ...curSelectedBox,
              class_name: targetName,
              class_id: keyIndex,
            });
            showToast(`Class: ${targetName.toUpperCase()} [#${curSelectedBox.id.slice(-4)}]`, 'label');
          } else {
            showToast(`Default Class: ${targetName.toUpperCase()}`, 'label');
          }
        }
      }

      // 7. AI Detection (R / Ctrl+Enter)
      else if (e.key === 'r' || e.key === 'R' || (e.ctrlKey && e.key === 'Enter')) {
        if (!isDetectingRef.current) {
          showToast('Running YOLO AI Detection... (R)', 'auto_awesome');
          handleRunYoloDetect();
        }
      }

      // 8. Save Annotations (Ctrl+S / Cmd+S)
      else if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S')) {
        e.preventDefault();
        if (curImg) {
          saveAnnotations(curImg.filename, curBoxes)
            .then(() => showToast(`Saved ${curBoxes.length} annotations to disk (Ctrl+S)`, 'save'))
            .catch((err) => showToast(`Save error: ${err}`, 'error'));
        }
      }

      // 9. Modals: Auto-Label AI (L / M), Import (I), Export (E), Help (?)
      else if (e.key === 'l' || e.key === 'L' || e.key === 'm' || e.key === 'M') {
        setIsAutoLabelOpen((prev) => !prev);
      } else if (e.key === 'i' || e.key === 'I') {
        setDatasetHubTab('import');
        setIsDatasetHubOpen(true);
      } else if (e.key === 'e' || e.key === 'E') {
        setDatasetHubTab('export');
        setIsDatasetHubOpen(true);
      } else if (e.key === '?' || e.key === 'F1') {
        setIsShortcutsOpen((prev) => !prev);
      }

      // 10. Escape: Close modals or deselect box
      else if (e.key === 'Escape') {
        if (isShortcutsOpenRef.current) setIsShortcutsOpen(false);
        else if (isAutoLabelOpenRef.current) setIsAutoLabelOpen(false);
        else if (isDatasetHubOpenRef.current) setIsDatasetHubOpen(false);
        else if (selectedBoxIdRef.current) {
          setSelectedBoxId(null);
          showToast('Deselected active box', 'close');
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showToast]);

  return (
    <div className="flex flex-col w-screen h-screen overflow-hidden bg-[#0a0e1a] font-sans relative">
      {/* 1. Global Studio Header */}
      <StudioHeader
        health={health}
        imageCount={images.length}
        onOpenAutoLabel={() => setIsAutoLabelOpen(true)}
        onOpenImport={() => {
          setDatasetHubTab('import');
          setIsDatasetHubOpen(true);
        }}
        onOpenExport={() => {
          setDatasetHubTab('export');
          setIsDatasetHubOpen(true);
        }}
        onOpenShortcuts={() => setIsShortcutsOpen(true)}
      />

      {/* 2. Main Workstation Body */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Leftmost Activity Tool Rail (44px) */}
        <ActivityRail activeTool={activeTool} onSelectTool={setActiveTool} />

        {/* Left Image Queue Drawer (215px) */}
        <ImageQueueDrawer
          images={images}
          currentImageIndex={currentImageIndex}
          onSelectImage={setCurrentImageIndex}
        />

        {/* Expansive Central Interactive Canvas (>80% width) */}
        <AnnotationCanvas
          image={currentImage}
          boxes={boxes}
          selectedBoxId={selectedBoxId}
          activeTool={activeTool}
          imageIndex={currentImageIndex}
          totalImages={images.length}
          activeClassName={activeClassName}
          availableClasses={health?.classes || []}
          readOnly={loadedImageKey !== currentImageKey}
          onSelectBox={setSelectedBoxId}
          onAddBox={handleAddBox}
          onUpdateBox={handleUpdateBox}
          onPrevImage={() => setCurrentImageIndex((prev) => Math.max(0, prev - 1))}
          onNextImage={() => setCurrentImageIndex((prev) => Math.min(images.length - 1, prev + 1))}
          onRunYoloDetect={handleRunYoloDetect}
          isDetecting={isDetecting}
        />

        {/* Right Compact Object Inspector (245px) */}
        <ObjectInspector
          selectedBox={selectedBox}
          availableClasses={health?.classes}
          onRegisterClass={async name => {
            const image = currentImageRef.current;
            const selected = selectedBoxRef.current;
            if (!image || !selected) throw new Error('Select an annotation first');
            const classes = await registerDatasetClass(name, image.dataset_id);
            if (imageIdentity(currentImageRef.current || image) !== imageIdentity(image)
                || selectedBoxRef.current !== selected) throw new Error('Selection changed; apply the class again');
            setHealth(previous => previous && previous.dataset_dir === image.dataset_id ? { ...previous, classes } : previous);
            return classes.findIndex(c => c.toLowerCase() === name.toLowerCase());
          }}
          onUpdateBox={handleUpdateBox}
          onDeleteBox={handleDeleteBox}
          onAcceptBox={handleAcceptBox}
        />
      </div>

      {saveError && <div role="alert" className="fixed top-12 left-1/2 -translate-x-1/2 z-50 bg-red-950 text-white p-3 rounded">
        {saveError.message}
        <button className="ml-3 underline" onClick={() => {
          const draft = draftsRef.current.get(imageIdentity(saveError.image));
          if (draft) void persistBoxes(draft, saveError.image);
        }}>Retry save</button>
      </div>}
      {/* Dynamic HUD Toast Notification */}
      {hudToast && (
        <div className="fixed bottom-14 left-1/2 -translate-x-1/2 z-50 pointer-events-none transition-all animate-fadeIn">
          <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-[#090f1d]/95 backdrop-blur-xl border border-[#4cd7f6]/50 text-[#dae2fd] text-xs font-mono shadow-2xl">
            <span className="material-symbols-outlined text-[#4cd7f6] text-[16px]">
              {hudToast.icon}
            </span>
            <span className="font-semibold tracking-wide">{hudToast.message}</span>
          </div>
        </div>
      )}

      {/* Keyboard Shortcuts Reference Modal */}
      <KeyboardShortcutsModal
        isOpen={isShortcutsOpen}
        onClose={() => setIsShortcutsOpen(false)}
      />

      {/* Auto-Label AI Modal (Screen 2) */}
      <AutoLabelModal
        isOpen={isAutoLabelOpen}
        onClose={() => setIsAutoLabelOpen(false)}
        health={health}
        totalImages={images.length}
        images={images}
        onStartBatch={() => { void refreshImages(); setAnnotationReload(value => value + 1); }}
        onBatchComplete={() => { void refreshImages(); setAnnotationReload(value => value + 1); }}
      />

      {/* Dataset Hub Modal (Screen 3) */}
      <DatasetHubModal
        isOpen={isDatasetHubOpen}
        onClose={() => setIsDatasetHubOpen(false)}
        totalImages={images.length}
        initialTab={datasetHubTab}
        onDatasetChanged={() => {
          setImages([]); setCurrentImageIndex(0); setBoxes([]); setLoadedImageKey('');
          void refreshImages();
        }}
      />
    </div>
  );
};

export default App;
