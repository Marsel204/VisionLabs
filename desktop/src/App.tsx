import React, { useState, useEffect, useCallback } from 'react';
import type { BoundingBox, ImageMeta, SystemHealth, ToolType } from './types';
import {
  fetchHealth,
  fetchImages,
  fetchAllImages,
  fetchAnnotations,
  saveAnnotations,
  detectYolo,
} from './services/api';
import { StudioHeader } from './components/studio/StudioHeader';
import { ActivityRail } from './components/studio/ActivityRail';
import { ImageQueueDrawer } from './components/studio/ImageQueueDrawer';
import { AnnotationCanvas } from './components/studio/AnnotationCanvas';
import { ObjectInspector } from './components/studio/ObjectInspector';
import { AutoLabelModal } from './components/modals/AutoLabelModal';
import { DatasetHubModal } from './components/modals/DatasetHubModal';

export const App: React.FC = () => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [images, setImages] = useState<ImageMeta[]>([]);
  const [currentImageIndex, setCurrentImageIndex] = useState<number>(0);
  const [boxes, setBoxes] = useState<BoundingBox[]>([]);
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null);
  const [activeTool, setActiveTool] = useState<ToolType>('bbox');

  const urlParams = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null;
  const [isAutoLabelOpen, setIsAutoLabelOpen] = useState(urlParams?.get('modal') === 'autolabel');
  const [isDatasetHubOpen, setIsDatasetHubOpen] = useState(urlParams?.get('modal') === 'dataset');
  const [datasetHubTab, setDatasetHubTab] = useState<'export' | 'import'>('import');
  const [isDetecting, setIsDetecting] = useState(false);

  const currentImage = images[currentImageIndex] || null;
  const selectedBox = boxes.find((b) => b.id === selectedBoxId) || null;

  /** Progressive refresh: show first 200 instantly, then load all in background */
  const refreshImages = useCallback(async () => {
    try {
      const first = await fetchImages(200, 0);
      setImages(first.images);
      // If there are more, load them all in background
      if (first.total > 200) {
        fetchAllImages()
          .then((all) => setImages(all.images))
          .catch((err) => console.error('Background image load failed:', err));
      }
    } catch (err) {
      console.error('Error refreshing images:', err);
    }
  }, []);

  // Initialize data from API — progressive load for instant first paint
  useEffect(() => {
    const init = async () => {
      try {
        const h = await fetchHealth();
        setHealth(h);
      } catch (err) {
        console.error('API health check error:', err);
      }

      try {
        // Phase 1: First 200 images → visible immediately (~140ms)
        const first = await fetchImages(200, 0);
        setImages(first.images);
        // Phase 2: Load all remaining images in background
        if (first.total > 200) {
          fetchAllImages()
            .then((all) => setImages(all.images))
            .catch((err) => console.error('Background image load failed:', err));
        }
      } catch (err) {
        console.error('Error fetching images:', err);
      }
    };
    init();
  }, []);


  // Load annotations when current image changes (do NOT auto-run YOLO — explicit user action only)
  useEffect(() => {
    if (!currentImage) return;
    const load = async () => {
      try {
        const data = await fetchAnnotations(currentImage.filename);
        if (data.boxes && data.boxes.length > 0) {
          setBoxes(data.boxes);
          setSelectedBoxId(data.boxes[0].id);
        } else {
          setBoxes([]);
          setSelectedBoxId(null);
        }
      } catch (err) {
        console.error('Error loading annotations:', err);
        setBoxes([]);
        setSelectedBoxId(null);
      }
    };
    load();
  }, [currentImage?.filename]);

  // Save annotations helper
  const persistBoxes = useCallback(
    async (updatedBoxes: BoundingBox[]) => {
      setBoxes(updatedBoxes);
      if (currentImage) {
        try {
          await saveAnnotations(currentImage.filename, updatedBoxes);
          setImages((prev) =>
            prev.map((img, idx) =>
              idx === currentImageIndex
                ? { ...img, annotation_count: updatedBoxes.length, status: 'reviewed' }
                : img
            )
          );
        } catch (err) {
          console.error('Failed to save annotations:', err);
        }
      }
    },
    [currentImage, currentImageIndex]
  );

  const handleAddBox = (newBox: BoundingBox) => {
    const updated = [...boxes, newBox];
    persistBoxes(updated);
  };

  const handleUpdateBox = (updatedBox: BoundingBox) => {
    const updated = boxes.map((b) => (b.id === updatedBox.id ? updatedBox : b));
    persistBoxes(updated);
  };

  const handleDeleteBox = (id: string) => {
    const updated = boxes.filter((b) => b.id !== id);
    if (selectedBoxId === id) {
      setSelectedBoxId(updated[0]?.id || null);
    }
    persistBoxes(updated);
  };

  const handleAcceptBox = (id: string) => {
    const box = boxes.find((b) => b.id === id);
    if (box) {
      handleUpdateBox({ ...box, confidence: 1.0 });
    }
  };

  const handleRunYoloDetect = async () => {
    if (!currentImage) return;
    setIsDetecting(true);
    try {
      const res = await detectYolo(currentImage.filename, 0.25);
      persistBoxes(res.boxes);
      setSelectedBoxId(res.boxes[0]?.id || null);
    } catch (err) {
      console.error('YOLO detection failed:', err);
    } finally {
      setIsDetecting(false);
    }
  };

  // Keyboard navigation shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in an input
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)) {
        return;
      }

      if (e.key === 'v' || e.key === 'V') setActiveTool('select');
      if (e.key === 'b' || e.key === 'B') setActiveTool('bbox');
      if (e.key === 'h' || e.key === 'H') setActiveTool('pan');
      if (e.key === 'ArrowLeft' || e.key === '[') {
        setCurrentImageIndex((prev) => Math.max(0, prev - 1));
      }
      if (e.key === 'ArrowRight' || e.key === ']') {
        setCurrentImageIndex((prev) => Math.min(images.length - 1, prev + 1));
      }
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedBoxId) {
        handleDeleteBox(selectedBoxId);
      }
      if (e.key === 'Escape') {
        setSelectedBoxId(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [images.length, selectedBoxId]);

  return (
    <div className="flex flex-col w-screen h-screen overflow-hidden bg-[#0a0e1a] font-sans">
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
          onUpdateBox={handleUpdateBox}
          onDeleteBox={handleDeleteBox}
          onAcceptBox={handleAcceptBox}
        />
      </div>

      {/* Auto-Label AI Modal (Screen 2) */}
      <AutoLabelModal
        isOpen={isAutoLabelOpen}
        onClose={() => setIsAutoLabelOpen(false)}
        health={health}
        totalImages={images.length}
        images={images}
        onStartBatch={(_classes, _conf) => {
          refreshImages();
          handleRunYoloDetect();
        }}
        onBatchComplete={refreshImages}
      />

      {/* Dataset Hub Modal (Screen 3) */}
      <DatasetHubModal
        isOpen={isDatasetHubOpen}
        onClose={() => setIsDatasetHubOpen(false)}
        totalImages={images.length}
        initialTab={datasetHubTab}
        onDatasetChanged={refreshImages}
      />
    </div>
  );
};

export default App;
