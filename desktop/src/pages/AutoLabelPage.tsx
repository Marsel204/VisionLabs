import React, { useState, useEffect, useCallback, useMemo } from 'react';
import type { ImageMeta, ModelPreparationStatus, Page, SystemHealth } from '../types';
import { getClassColor } from '../types';
import {
  autoRefinePrompt,
  runAutoLabelPreview,
  startAutoLabelBatch,
  fetchAutoLabelStatus,
  fetchModelPresets,
  prepareAutoLabelModels,
  fetchModelPreparationStatus,
  browseCustomWeights,
  validateCustomModel,
  getImageUrl,
} from '../services/api';
import { Badge, Button, Card, Icon, PageHeader, ProgressBar } from '../components/ui';
import { formatCount } from '../format';
import { ImageWithBoxes } from '../components/ImageWithBoxes';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onStartBatch: (classes: string[], conf: number) => void;
  health?: SystemHealth | null;
  totalImages?: number;
  /** Images that are neither reviewed nor auto-labeled: what a batch will process. */
  pendingImages?: number;
  images?: ImageMeta[];
  onBatchComplete?: () => void;
  onNavigate?: (page: Page) => void;
  onOpenColab?: () => void;
}

interface SemanticClass {
  id: string;
  name: string;
  prompt: string;
  color: string;
  enabled: boolean;
}

export const AutoLabelPage: React.FC<Props> = ({
  isOpen,
  onClose,
  onStartBatch,
  health,
  totalImages = 0,
  pendingImages,
  images,
  onBatchComplete,
  onNavigate,
  onOpenColab,
}) => {
  // Detector & Verifier flags
  const [enableGroundingDino, setEnableGroundingDino] = useState(true);
  const [enableYolo, setEnableYolo] = useState(true);
  const [flipPass, setFlipPass] = useState(false);
  const [enableFlorenceDetector, setEnableFlorenceDetector] = useState(false);
  const [enableLocateAnything, setEnableLocateAnything] = useState(false);
  const [enableFlorenceVerifier, setEnableFlorenceVerifier] = useState(true);
  const [enableSam2Masks, setEnableSam2Masks] = useState(true);

  // Multi-Model YOLO & Custom Weights (up to 3 models simultaneously)
  const [activeYoloModels, setActiveYoloModels] = useState<string[]>(['yolo11n.pt']);
  const [presets, setPresets] = useState<Array<{ id: string; name: string; size: string; type: string }>>([
    { id: 'yolo11n.pt', name: 'YOLO11 Nano (Default)', size: '5.4 MB', type: 'preset' },
    { id: 'yolo11s.pt', name: 'YOLO11 Small', size: '19.0 MB', type: 'preset' },
    { id: 'yolo11m.pt', name: 'YOLO11 Medium', size: '40.2 MB', type: 'preset' },
    { id: 'yolov8n.pt', name: 'YOLOv8 Nano', size: '6.2 MB', type: 'preset' },
    { id: 'yolov8m.pt', name: 'YOLOv8 Medium', size: '49.7 MB', type: 'preset' },
    { id: 'yolov8x.pt', name: 'YOLOv8 Extra-Large', size: '136.7 MB', type: 'preset' },
  ]);
  const [showAddMenu, setShowAddMenu] = useState(false);
  const [replacingModelIdx, setReplacingModelIdx] = useState<number | null>(null);
  const [modelActionMessage, setModelActionMessage] = useState<string | null>(null);

  // Semantic class prompts
  const [classes, setClasses] = useState<SemanticClass[]>(() => {
    if (health?.classes && health.classes.length > 0) {
      return health.classes.map((cls, idx) => ({
        id: String(idx + 1).padStart(2, '0'),
        name: cls.charAt(0).toUpperCase() + cls.slice(1),
        prompt: `${cls}, visual entity`,
        color: getClassColor(cls).stroke,
        enabled: true,
      }));
    }
    return [
      {
        id: '01',
        name: 'Object',
        prompt: 'object, visual entity, item',
        color: '#06b6d4',
        enabled: true,
      },
    ];
  });

  useEffect(() => {
    if (health?.classes && health.classes.length > 0) {
      setClasses((prev) => {
        if (prev.length === 1 && prev[0].name === 'Object') {
          return health.classes.map((cls, idx) => ({
            id: String(idx + 1).padStart(2, '0'),
            name: cls.charAt(0).toUpperCase() + cls.slice(1),
            prompt: `${cls}, visual entity`,
            color: getClassColor(cls).stroke,
            enabled: true,
          }));
        }
        return prev;
      });
    }
  }, [health?.classes]);

  const hasValidClasses = classes.some(cls => cls.enabled && cls.name.trim()) &&
    classes.every(cls => !cls.enabled || cls.name.trim().length > 0);

  // Hyperparameters
  const [confidence, setConfidence] = useState<number>(0.25);
  const [iouThreshold, setIouThreshold] = useState<number>(0.45);
  const [strictVlm, setStrictVlm] = useState<boolean>(true);
  const [maxInstances, setMaxInstances] = useState<number>(50);

  // Auto-refining state
  const [refiningId, setRefiningId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);

  // Verification View Mode: 'both' | 'masks' | 'raw'
  const [viewMode, setViewMode] = useState<'both' | 'masks' | 'raw'>('both');

  // Resampling / re-scoring animation state
  const [isRescoring, setIsRescoring] = useState(false);
  const [isBatchRunning, setIsBatchRunning] = useState(false);
  const [batchProgress, setBatchProgress] = useState<import('../types').AutoLabelStatus | null>(null);
  const [liveMeanConf, setLiveMeanConf] = useState<number | null>(null);

  // Dataset image sampling
  const [sampleOffset, setSampleOffset] = useState<number>(0);
  const [previewDetections, setPreviewDetections] = useState<
    Record<
      string,
      {
        loading?: boolean;
        detections: any[];
        count: number;
        mean_confidence?: number;
        iou?: number;
        elapsed_seconds?: number;
        fallback?: boolean;
        error?: string;
      }
    >
  >({});

  const [modelPreparation, setModelPreparation] = useState<ModelPreparationStatus | null>(null);
  const [preparedSelection, setPreparedSelection] = useState('');
  const [modelPreparationError, setModelPreparationError] = useState<string | null>(null);
  const [preparationAttempt, setPreparationAttempt] = useState(0);
  const modelOptions = useMemo(() => ({
    enable_grounding_dino: enableGroundingDino,
    enable_sam2_masks: enableSam2Masks,
    enable_florence2: enableFlorenceDetector,
    enable_locate_anything: enableLocateAnything,
    enable_florence2_verifier: enableFlorenceVerifier,
    enable_yolo: enableYolo,
    yolo_models: activeYoloModels,
  }), [enableGroundingDino, enableSam2Masks, enableFlorenceDetector, enableLocateAnything,
    enableFlorenceVerifier, enableYolo, activeYoloModels]);
  const selectionKey = JSON.stringify(modelOptions);
  const selectedModels = [
    ...(enableGroundingDino ? [{ id: 'grounding_dino', name: 'Grounding DINO' }] : []),
    ...(enableYolo ? activeYoloModels.map(path => ({
      id: `yolo:${path}`, name: path.replaceAll('\\', '/').split('/').pop() || path,
    })) : []),
    ...(enableFlorenceDetector || enableFlorenceVerifier ? [{ id: 'florence2', name: 'Florence-2' }] : []),
    ...(enableLocateAnything ? [{ id: 'locate_anything', name: 'Locate Anything 3B' }] : []),
    ...(enableSam2Masks ? [{ id: 'sam2', name: 'SAM 2' }] : []),
  ].filter((model, index, models) => models.findIndex(candidate => candidate.id === model.id) === index);
  const hasDetector = enableGroundingDino || enableYolo || enableFlorenceDetector || enableLocateAnything;
  const isCloud = modelPreparation?.backend === 'cloud';
  const yoloViews = flipPass && !isCloud ? ['original', 'flip'] : ['original'];
  const readyModelCount = selectedModels.filter(model => modelPreparation?.models[model.id]?.state === 'ready').length;
  const preparationFailed = !!modelPreparationError || selectedModels.some(model =>
    modelPreparation?.models[model.id]?.state === 'error');
  const modelsReady = hasDetector && !preparationFailed && preparedSelection === selectionKey &&
    (isCloud || readyModelCount === selectedModels.length);
  const modelStatusText = !hasDetector ? 'Pilih detektor objek'
    : preparationFailed ? 'Persiapan model gagal'
    : modelsReady ? (isCloud ? 'Inference cloud siap' : 'Model siap untuk auto-label')
    : 'Mengunduh / memuat model…';

  useEffect(() => {
    if (!isOpen || !hasDetector) return;
    let active = true;
    let polling = false;
    let interval: ReturnType<typeof setInterval> | undefined;
    const applyStatus = (status: ModelPreparationStatus) => {
      if (!active) return;
      setModelPreparation(status);
      setPreparedSelection(selectionKey);
      setModelPreparationError(null);
    };
    const poll = async () => {
      if (polling) return;
      polling = true;
      try {
        applyStatus(await fetchModelPreparationStatus());
      } catch (error) {
        if (active) setModelPreparationError(`Status unduhan model tidak bisa dicek: ${String(error)}`);
      } finally {
        polling = false;
      }
    };
    prepareAutoLabelModels(modelOptions).then(status => {
      if (!active) return;
      applyStatus(status);
      interval = setInterval(poll, 750);
    }).catch(error => {
      if (active) setModelPreparationError(`Model tidak bisa disiapkan: ${String(error)}`);
    });
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [isOpen, hasDetector, modelOptions, selectionKey, preparationAttempt]);

  const sampleImages = useMemo(() => {
    if (!images || images.length === 0) return [];
    const slice = images.slice(sampleOffset, sampleOffset + 4);
    if (slice.length === 0) return images.slice(0, 4);
    return slice;
  }, [images, sampleOffset]);

  const computePipelineMode = useCallback((): string => {
    const activeDetectorsCount =
      (enableGroundingDino ? 1 : 0) +
      (enableYolo ? 1 : 0) +
      (enableFlorenceDetector ? 1 : 0) +
      (enableLocateAnything ? 1 : 0);

    if (activeDetectorsCount > 1 || (enableYolo && activeYoloModels.length > 1)) {
      return enableSam2Masks ? 'ensemble_fusion_sam2_masks' : 'ensemble_fusion_boxes';
    }
    if (enableYolo) {
      return enableSam2Masks ? 'yolo_sam2_masks' : 'yolo_boxes';
    }
    if (enableFlorenceDetector) {
      return enableSam2Masks ? 'vlm_sam2_masks' : 'vlm_boxes';
    }
    if (enableLocateAnything) {
      return enableSam2Masks ? 'locate_anything_sam2_masks' : 'locate_anything_boxes';
    }
    return enableSam2Masks ? 'sam2_dino_masks' : 'dino_boxes';
  }, [
    enableGroundingDino,
    enableYolo,
    enableFlorenceDetector,
    enableLocateAnything,
    activeYoloModels.length,
    enableSam2Masks,
  ]);

  const fetchPreviewForImages = useCallback(
    async (imgsToPreview: ImageMeta[]) => {
      if (!modelsReady || !hasValidClasses || imgsToPreview.length === 0 || isRescoring || isBatchRunning) return;
      setIsRescoring(true);
      try {
        const results = await Promise.allSettled(
          imgsToPreview.map(async (img) => {
            const target = img.path || img.filename;
            const res = await runAutoLabelPreview({
              image_name: target,
              dataset_id: img.dataset_id,
              classes,
              confidence_threshold: confidence,
              iou_threshold: iouThreshold,
              strict_vlm: strictVlm,
              max_instances: maxInstances,
              pipeline_mode: computePipelineMode(),
              enable_grounding_dino: enableGroundingDino,
              enable_sam2_masks: enableSam2Masks,
              enable_florence2: enableFlorenceDetector,
              enable_locate_anything: enableLocateAnything,
              enable_florence2_verifier: enableFlorenceVerifier,
              enable_yolo: enableYolo,
              yolo_models: activeYoloModels,
              yolo_views: yoloViews,
            });
            return { filename: img.filename, path: img.path, data: res };
          })
        );

        const newDets: typeof previewDetections = {};
        let totalConf = 0;
        let confCount = 0;

        results.forEach((r, idx) => {
          const filename = imgsToPreview[idx].filename;
          const imgPath = imgsToPreview[idx].path;
          if (r.status === 'fulfilled') {
            const entry = {
              loading: false,
              detections: r.value.data.detections || [],
              count: r.value.data.count || 0,
              mean_confidence: r.value.data.mean_confidence,
              iou: r.value.data.iou,
              elapsed_seconds: r.value.data.elapsed_seconds,
              fallback: r.value.data.fallback,
            };
            newDets[filename] = entry;
            if (imgPath) newDets[imgPath] = entry;
            if (r.value.data.mean_confidence) {
              totalConf += r.value.data.mean_confidence;
              confCount++;
            }
          } else {
            const errEntry = {
              loading: false,
              detections: [],
              count: 0,
              error: String(r.reason),
            };
            newDets[filename] = errEntry;
            if (imgPath) newDets[imgPath] = errEntry;
          }
        });

        setPreviewDetections((prev) => ({ ...prev, ...newDets }));
        if (confCount > 0) {
          setLiveMeanConf(Number((totalConf / confCount).toFixed(3)));
        }
      } catch (err) {
        console.error('Error fetching preview detections:', err);
      } finally {
        setIsRescoring(false);
      }
    },
    [
      classes,
      confidence,
      iouThreshold,
      strictVlm,
      maxInstances,
      enableGroundingDino,
      enableSam2Masks,
      enableFlorenceDetector,
      enableLocateAnything,
      enableFlorenceVerifier,
      enableYolo,
      activeYoloModels,
      yoloViews,
      computePipelineMode,
      modelsReady,
      hasValidClasses,
      isRescoring,
      isBatchRunning,
    ]
  );

  useEffect(() => {
    if (isOpen) {
      fetchModelPresets()
        .then((res) => {
          if (res.presets && res.presets.length > 0) setPresets(res.presets);
        })
        .catch(() => {});
    }
  }, [isOpen]);

  useEffect(() => {
    if (isOpen && modelsReady && hasValidClasses && !isBatchRunning && sampleImages.length > 0) {
      const timer = setTimeout(() => {
        fetchPreviewForImages(sampleImages);
      }, 350);
      return () => clearTimeout(timer);
    }
  }, [
    isOpen,
    modelsReady,
    hasValidClasses,
    isBatchRunning,
    sampleImages.map((s) => s.path || s.filename).join(','),
    confidence,
    iouThreshold,
    enableGroundingDino,
    enableSam2Masks,
    enableFlorenceDetector,
    enableLocateAnything,
    enableFlorenceVerifier,
    enableYolo,
    activeYoloModels.join(','),
    yoloViews.join(','),
  ]);

  if (!isOpen) return null;

  const handleBrowseCustomWeights = async (replaceIdx?: number) => {
    try {
      const res = await browseCustomWeights();
      if (res.status === 'ok' && res.path) {
        try {
          const valid = await validateCustomModel(res.path);
          if (typeof replaceIdx === 'number' && replaceIdx >= 0 && replaceIdx < activeYoloModels.length) {
            setActiveYoloModels((prev) => {
              const updated = [...prev];
              updated[replaceIdx] = valid.path;
              return updated;
            });
            setModelActionMessage(`Bobot kustom dimuat: ${valid.name}`);
          } else if (activeYoloModels.length < 3) {
            setActiveYoloModels((prev) => [...prev, valid.path]);
            setEnableYolo(true);
            setModelActionMessage(`Model kustom ditambahkan: ${valid.name}`);
          }
        } catch {
          if (typeof replaceIdx === 'number' && replaceIdx >= 0) {
            setActiveYoloModels((prev) => {
              const updated = [...prev];
              updated[replaceIdx] = res.path!;
              return updated;
            });
          } else if (activeYoloModels.length < 3) {
            setActiveYoloModels((prev) => [...prev, res.path!]);
            setEnableYolo(true);
          }
          setModelActionMessage(`Bobot dimuat: ${res.name || res.path}`);
        }
      }
    } catch (err) {
      setModelActionMessage(`Gagal memilih file: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setShowAddMenu(false);
      setReplacingModelIdx(null);
    }
  };

  const handleSelectPreset = (presetId: string, replaceIdx?: number) => {
    if (typeof replaceIdx === 'number' && replaceIdx >= 0 && replaceIdx < activeYoloModels.length) {
      setActiveYoloModels((prev) => {
        const updated = [...prev];
        updated[replaceIdx] = presetId;
        return updated;
      });
      setModelActionMessage(`Model diganti ke preset: ${presetId}`);
    } else if (activeYoloModels.length < 3 && !activeYoloModels.includes(presetId)) {
      setActiveYoloModels((prev) => [...prev, presetId]);
      setEnableYolo(true);
      setModelActionMessage(`Preset ditambahkan: ${presetId}`);
    }
    setShowAddMenu(false);
    setReplacingModelIdx(null);
  };

  const handleRemoveModel = (idx: number) => {
    if (activeYoloModels.length > 1) {
      setActiveYoloModels((prev) => prev.filter((_, i) => i !== idx));
    }
  };

  const handleAutoRefine = async (cls: SemanticClass) => {
    setRefiningId(cls.id);
    try {
      const res = await autoRefinePrompt(cls.name, cls.prompt);
      if (res.refined_prompt) {
        setClasses((prev) =>
          prev.map((c) =>
            c.id === cls.id ? { ...c, prompt: res.refined_prompt } : c
          )
        );
      }
    } catch (err) {
      console.error('Failed to auto-refine prompt:', err);
    } finally {
      setRefiningId(null);
    }
  };

  const handleDeleteClass = (id: string) => {
    setClasses((prev) => prev.filter((c) => c.id !== id));
  };

  const handleRenameClass = (id: string, name: string) => {
    setClasses(prev => prev.map(cls => {
      if (cls.id !== id) return cls;
      const previousName = cls.name.trim();
      const usesDefaultPrompt = cls.prompt === `${previousName}, visual entity` ||
        cls.prompt === `${previousName.toLowerCase()}, visual entity` ||
        cls.prompt === 'new custom visual object description' ||
        (previousName === 'Object' && cls.prompt === 'object, visual entity, item');
      return { ...cls, name, prompt: usesDefaultPrompt ? `${name.trim()}, visual entity` : cls.prompt };
    }));
  };

  const handleAddClass = () => {
    const nextNum = Math.max(0, ...classes.map(cls => Number(cls.id) || 0)) + 1;
    const newId = String(nextNum).padStart(2, '0');
    const colors = ['#ec4899', '#3b82f6', '#14b8a6', '#f97316', '#8b5cf6'];
    const chosenColor = colors[classes.length % colors.length];
    setClasses((prev) => [
      ...prev,
      {
        id: newId,
        name: `Class_${newId}`,
        prompt: `Class_${newId}, visual entity`,
        color: chosenColor,
        enabled: true,
      },
    ]);
    setEditingId(newId);
  };

  const handleRescore = () => {
    if (sampleImages.length > 0) {
      fetchPreviewForImages(sampleImages);
    }
  };

  const handleResampleFrames = () => {
    if (images && images.length > 0) {
      setSampleOffset((prev) => (prev + 4) % images.length);
    }
  };

  const handleLaunchBatch = async () => {
    if (!modelsReady || !hasValidClasses || isRescoring || isBatchRunning) return;
    setIsBatchRunning(true);
    try {
      await startAutoLabelBatch({
        dataset_id: health?.dataset_dir,
        classes,
        confidence_threshold: confidence,
        iou_threshold: iouThreshold,
        pipeline_mode: computePipelineMode(),
        only_unannotated: true,
        enable_grounding_dino: enableGroundingDino,
        enable_sam2_masks: enableSam2Masks,
        enable_florence2: enableFlorenceDetector,
        enable_locate_anything: enableLocateAnything,
        enable_florence2_verifier: enableFlorenceVerifier,
        enable_yolo: enableYolo,
        yolo_models: activeYoloModels,
        yolo_views: yoloViews,
      });

      // Poll status until done
      const interval = setInterval(async () => {
        try {
          const status = await fetchAutoLabelStatus();
          setBatchProgress(status);
          if (status.completed || !status.running) {
            clearInterval(interval);
            setIsBatchRunning(false);
            const activeClassNames = classes.filter((c) => c.enabled).map((c) => c.name);
            if (status.error || status.warnings?.length) {
              setModelActionMessage(status.error || status.warnings!.join('; ')); onBatchComplete?.(); return;
            }
            onStartBatch(activeClassNames, confidence);
            onBatchComplete?.();
            onClose();
          }
        } catch (pollErr) {
          setModelActionMessage(`Status batch tidak tersedia: ${String(pollErr)}`);
          clearInterval(interval);
          setIsBatchRunning(false);
        }
      }, 500);
    } catch (err) {
      setModelActionMessage(`Batch tidak bisa dimulai: ${String(err)}`);
      setIsBatchRunning(false);
    }
  };

  const activePipelineSteps: string[] = [];
  if (enableGroundingDino) activePipelineSteps.push('Grounding DINO');
  if (enableYolo) {
    activePipelineSteps.push(
      activeYoloModels.length > 1 ? `YOLO ×${activeYoloModels.length}` : 'YOLO'
    );
  }
  if (enableFlorenceDetector) activePipelineSteps.push('Florence-2');
  if (enableLocateAnything) activePipelineSteps.push('Locate Anything 3B');
  if (enableFlorenceVerifier) activePipelineSteps.push('Verifikasi Florence-2');
  if (enableSam2Masks) activePipelineSteps.push('Mask SAM 2');

  const executionPipelineStr =
    activePipelineSteps.length > 0 ? activePipelineSteps.join(' → ') : 'Belum dipilih';


  const batchPercent = batchProgress?.total
    ? Math.round(((batchProgress.current || 0) / Math.max(1, batchProgress.total)) * 100) : 0;
  const batchTarget = pendingImages ?? totalImages;
  const showJob = isBatchRunning || !!batchProgress;

  /** A selectable model card in the style of the design's "Pilih model AI" list. */
  const modelCard = (opts: {
    active: boolean; onToggle: () => void; title: string; description: string; badge?: string; children?: React.ReactNode;
  }) => (
    <div className={`rounded-xl border transition-colors ${
      opts.active ? 'border-brand-600 bg-brand-50 ring-1 ring-brand-600' : 'border-line bg-white hover:border-brand-300'
    }`}>
      <div
        role="checkbox"
        aria-checked={opts.active}
        tabIndex={0}
        onClick={opts.onToggle}
        onKeyDown={event => { if (event.key === ' ' || event.key === 'Enter') { event.preventDefault(); opts.onToggle(); } }}
        className="flex cursor-pointer items-start gap-3 px-4 py-3.5"
      >
        <Icon name={opts.active ? 'check_circle' : 'stacks'} className={`mt-0.5 text-[20px] ${opts.active ? 'text-brand-600' : 'text-ink-400'}`} />
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-semibold text-ink-900">{opts.title}</span>
          <span className="mt-1 block text-xs text-ink-500">{opts.description}</span>
        </span>
        {opts.badge && <Badge tone="brand" className="shrink-0">{opts.badge}</Badge>}
      </div>
      {opts.children}
    </div>
  );

  return (
    <div className="mx-auto max-w-[1040px] px-8 py-10">
      <PageHeader
        title="Mulai anotasi dengan AI"
        subtitle="Pilih model, cek pratinjau pada beberapa gambar, lalu jalankan batch pada gambar yang belum dianotasi."
        actions={onOpenColab && <Button variant="secondary" icon="cloud" onClick={onOpenColab}>Jalankan di Colab</Button>}
      />

      <section
        aria-label="Model download status"
        role="status"
        aria-live="polite"
        className={`mb-6 rounded-xl border px-5 py-4 ${preparationFailed ? 'border-danger-600/30 bg-danger-50' : 'border-line bg-brand-50'}`}
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-sm font-semibold text-ink-900">
            <Icon
              name={preparationFailed ? 'error' : modelsReady ? 'check_circle' : hasDetector ? 'progress_activity' : 'info'}
              className={`text-[20px] ${preparationFailed ? 'text-danger-600' : modelsReady ? 'text-ok-600' : 'text-brand-600'} ${
                !preparationFailed && !modelsReady && hasDetector ? 'motion-safe:animate-spin' : ''}`}
            />
            <span>{modelStatusText}</span>
          </div>
          {hasDetector && !isCloud && (
            <span className="text-xs tabular-nums text-ink-500">{readyModelCount}/{selectedModels.length} model siap</span>
          )}
        </div>
        {!hasDetector ? (
          <p className="mt-1 text-xs text-ink-500">Aktifkan minimal satu detektor objek untuk melanjutkan.</p>
        ) : preparationFailed ? (
          <div className="mt-2 flex flex-wrap items-start justify-between gap-2">
            <div className="min-w-0 flex-1 break-words text-xs text-danger-600">
              <p>{modelPreparationError || 'Model yang dipilih gagal diunduh atau dimuat. Cek koneksi lalu coba lagi, atau nonaktifkan model itu.'}</p>
              {selectedModels.filter(model => modelPreparation?.models[model.id]?.state === 'error').map(model => (
                <p key={model.id} className="mt-1">{model.name}: {modelPreparation?.models[model.id]?.error}</p>
              ))}
            </div>
            <Button size="sm" variant="danger" onClick={() => setPreparationAttempt(attempt => attempt + 1)}>Coba siapkan ulang</Button>
          </div>
        ) : !modelsReady ? (
          <p className="mt-1 text-xs text-ink-500">
            Model yang belum ada diunduh otomatis. Pemakaian pertama bisa makan beberapa menit. Pratinjau dan batch terbuka setelah semua model siap.
          </p>
        ) : isCloud ? (
          <p className="mt-1 text-xs text-ink-500">Model yang dipilih berjalan di worker cloud-mu. Tidak perlu mengunduh model lokal.</p>
        ) : null}
        {hasDetector && !isCloud && (
          <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-500">
            {selectedModels.map(model => {
              const state = modelPreparation?.models[model.id]?.state;
              return <li key={model.id} className="min-w-0 break-all">
                {model.name}: <span className={state === 'ready' ? 'text-ok-600' : state === 'error' ? 'text-danger-600' : 'text-brand-600'}>
                  {state === 'ready' ? 'Siap' : state === 'error' ? 'Gagal' : state === 'preparing' ? 'Mengunduh / memuat…' : state === 'queued' ? 'Menunggu' : 'Mengecek…'}
                </span>
              </li>;
            })}
          </ul>
        )}
      </section>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_352px]">
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-ink-900">Pilih model AI</h2>
            <Badge>Bisa digabung</Badge>
          </div>

          <div className="mt-5 space-y-3">
            {modelCard({
              active: enableYolo, onToggle: () => setEnableYolo(!enableYolo), title: 'YOLO',
              description: 'Deteksi kendaraan cepat untuk batch besar. Bisa memakai bobot kustom (.pt / .onnx).',
              badge: 'Direkomendasikan',
              children: enableYolo && (
                <div className="space-y-2 border-t border-brand-200 px-4 pb-4 pt-3">
                  <div className="flex items-center justify-between text-[11px] text-ink-500">
                    <span>Bobot aktif ({activeYoloModels.length}/3)</span>
                    {activeYoloModels.length < 3 && (
                      <button type="button" onClick={() => { setReplacingModelIdx(null); setShowAddMenu(!showAddMenu); }}
                        className="flex items-center gap-1 font-semibold text-brand-600 hover:text-brand-700 cursor-pointer">
                        <Icon name="add" className="text-[15px]" /> Tambah model
                      </button>
                    )}
                  </div>
                  {activeYoloModels.map((modelPath, idx) => {
                    const isPreset = !modelPath.includes('/') && !modelPath.includes('\\');
                    const displayName = modelPath.split(/[\\/]/).filter(Boolean).pop() || modelPath;
                    const ready = modelPreparation?.models[`yolo:${modelPath}`]?.state === 'ready';
                    return (
                      <div key={`model-${idx}-${modelPath}`} className="flex items-center gap-2 rounded-lg border border-line bg-white px-3 py-2 text-xs">
                        <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${ready ? 'bg-ok-600' : 'bg-ink-300'}`} />
                        <span className="min-w-0 flex-1 truncate font-medium text-ink-900" title={modelPath}>{displayName}</span>
                        <Badge tone={isPreset ? 'brand' : 'neutral'} className="!py-0.5 text-[10px]">{isPreset ? 'Preset' : 'Kustom'}</Badge>
                        <button type="button" onClick={() => { setReplacingModelIdx(idx); setShowAddMenu(true); }}
                          className="text-[11px] font-semibold text-brand-600 hover:text-brand-700 cursor-pointer" title="Ganti bobot model">
                          Ganti
                        </button>
                        {activeYoloModels.length > 1 && (
                          <button type="button" onClick={() => handleRemoveModel(idx)} title="Hapus model dari gabungan"
                            className="flex h-6 w-6 items-center justify-center rounded text-ink-400 hover:bg-danger-50 hover:text-danger-600 cursor-pointer">
                            <Icon name="close" className="text-[15px]" />
                          </button>
                        )}
                      </div>
                    );
                  })}
                  {showAddMenu && (
                    <div className="space-y-2 rounded-lg border border-brand-200 bg-white p-3 shadow-sm">
                      <div className="flex items-center justify-between text-[11px] font-semibold text-ink-700">
                        <span>{replacingModelIdx !== null ? `Ganti model ${replacingModelIdx + 1} dengan:` : 'Tambah model ke gabungan:'}</span>
                        <button type="button" aria-label="Tutup" onClick={() => { setShowAddMenu(false); setReplacingModelIdx(null); }}
                          className="text-ink-400 hover:text-ink-900 cursor-pointer"><Icon name="close" className="text-[15px]" /></button>
                      </div>
                      <Button size="sm" variant="soft" icon="folder_open" block onClick={() => handleBrowseCustomWeights(replacingModelIdx ?? undefined)}>
                        Pilih bobot kustom (.pt, .onnx, .engine)…
                      </Button>
                      <div className="grid grid-cols-2 gap-1.5">
                        {presets.map((p) => (
                          <button key={p.id} type="button" onClick={() => handleSelectPreset(p.id, replacingModelIdx ?? undefined)}
                            className="flex items-center justify-between rounded-md border border-line px-2 py-1.5 text-left text-[11px] text-ink-700 hover:border-brand-300 hover:bg-brand-50 cursor-pointer">
                            <span className="truncate">{p.id}</span>
                            <span className="text-[10px] text-ink-400">{p.size}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  <label
                    htmlFor="yolo-flip-pass"
                    className={`flex items-start gap-2 pt-1 text-xs ${isCloud ? 'opacity-50' : 'cursor-pointer'}`}
                    title="Juga mendeteksi pada salinan cermin lalu menggabungkan hasilnya. Di data lalu lintas yang diuji, recall motor naik sekitar 3 sampai 7 poin, sedikit lebih baik daripada menurunkan ambang confidence, dengan waktu YOLO sekitar 2x dan lebih banyak kotak untuk direview. Hanya mesin lokal."
                  >
                    <input id="yolo-flip-pass" type="checkbox" checked={flipPass && !isCloud} disabled={isCloud}
                      onChange={(e) => setFlipPass(e.target.checked)} className="mt-0.5 h-3.5 w-3.5 accent-brand-600" />
                    <span>
                      <span className="font-semibold text-ink-900">Pass cermin: temukan lebih banyak objek tertutup</span>
                      <span className="block text-[11px] text-ink-500">
                        {isCloud ? 'Hanya untuk mesin lokal.' : 'Menggabungkan deteksi dari salinan cermin. Sekitar 2× lebih lambat.'}
                      </span>
                    </span>
                  </label>
                </div>
              ),
            })}
            {modelCard({
              active: enableGroundingDino, onToggle: () => setEnableGroundingDino(!enableGroundingDino),
              title: 'Grounding DINO', description: 'Deteksi objek dari deskripsi teks setiap kelas.',
            })}
            <label
              htmlFor="locate-anything-detector"
              className={`flex cursor-pointer items-start gap-3 rounded-xl border px-4 py-3.5 transition-colors ${
                enableLocateAnything ? 'border-brand-600 bg-brand-50 ring-1 ring-brand-600' : 'border-line bg-white hover:border-brand-300'
              }`}
            >
              <input
                id="locate-anything-detector"
                type="checkbox"
                checked={enableLocateAnything}
                onChange={(event) => setEnableLocateAnything(event.target.checked)}
                aria-describedby="locate-anything-description"
                className="mt-1 h-4 w-4 accent-brand-600"
              />
              <span>
                <span className="block text-sm font-semibold text-ink-900">Locate Anything 3B</span>
                <span id="locate-anything-description" className="mt-1 block text-xs text-ink-500">
                  Lokalisasi objek berbasis prompt dari NVIDIA. Diunduh saat pertama dipakai. Kotaknya tidak punya skor confidence.
                </span>
              </span>
            </label>
            {modelCard({
              active: enableFlorenceDetector, onToggle: () => setEnableFlorenceDetector(!enableFlorenceDetector),
              title: 'Florence-2', description: 'Deteksi dan deskripsi visual dengan model bahasa-visual.',
            })}
          </div>

          <h3 className="mt-6 text-[13px] font-semibold text-ink-900">Verifikasi &amp; mask</h3>
          <div className="mt-3 space-y-3">
            {modelCard({
              active: enableFlorenceVerifier, onToggle: () => setEnableFlorenceVerifier(!enableFlorenceVerifier),
              title: 'Verifikasi Florence-2', description: 'Memeriksa ulang potongan setiap kotak untuk membuang deteksi palsu.',
            })}
            {modelCard({
              active: enableSam2Masks,
              onToggle: () => {
                const next = !enableSam2Masks;
                setEnableSam2Masks(next);
                if (!next && viewMode === 'masks') setViewMode('both');
              },
              title: 'SAM 2', description: 'Segmentasi objek dengan mask poligon detail dari setiap kotak.',
            })}
          </div>

          {modelActionMessage && (
            <div className="mt-4 flex items-center justify-between gap-2 rounded-lg bg-brand-50 px-3 py-2 text-xs text-ink-700">
              <span>{modelActionMessage}</span>
              <button type="button" aria-label="Tutup pesan" onClick={() => setModelActionMessage(null)} className="text-ink-400 hover:text-ink-900 cursor-pointer">
                <Icon name="close" className="text-[15px]" />
              </button>
            </div>
          )}

          <div className="mt-7 flex items-center justify-between">
            <h3 className="text-[13px] font-semibold text-ink-900">Kelas target</h3>
            <span className="text-xs text-ink-500">{classes.length} kelas</span>
          </div>
          <div className="mt-3 space-y-2">
            {classes.map((c) => {
              const isRefining = refiningId === c.id;
              const isEditing = editingId === c.id;
              return (
                <div key={c.id} className="rounded-lg border border-line p-3">
                  <div className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ backgroundColor: c.color }} />
                    <label htmlFor={`class-name-${c.id}`} className="sr-only">Nama kelas</label>
                    <input
                      id={`class-name-${c.id}`}
                      type="text"
                      value={c.name}
                      onChange={event => handleRenameClass(c.id, event.target.value)}
                      onBlur={event => handleRenameClass(c.id, event.target.value.trim())}
                      onFocus={event => event.currentTarget.select()}
                      autoFocus={isEditing}
                      aria-invalid={!c.name.trim()}
                      aria-describedby={!c.name.trim() ? 'class-name-error' : undefined}
                      placeholder="mis. rambu lalu lintas"
                      className={`h-8 min-w-0 flex-1 rounded-md border px-2 text-sm font-semibold text-ink-900 focus:border-brand-500 focus:outline-none ${
                        c.name.trim() ? 'border-transparent hover:border-line-strong' : 'border-danger-600'
                      }`}
                    />
                    <button
                      onClick={() => handleAutoRefine(c)}
                      disabled={isRefining || !c.name.trim()}
                      title="Perbaiki prompt otomatis dengan Florence-2"
                      type="button"
                      className="flex h-7 items-center gap-1 rounded-md px-2 text-[11px] font-semibold text-brand-600 hover:bg-brand-50 cursor-pointer disabled:opacity-50"
                    >
                      <Icon name={isRefining ? 'progress_activity' : 'auto_fix_high'} className={`text-[15px] ${isRefining ? 'animate-spin' : ''}`} />
                      {isRefining ? 'Memperbaiki…' : 'Perbaiki'}
                    </button>
                    <button onClick={() => setEditingId(isEditing ? null : c.id)} title="Edit prompt text" aria-label="Ubah prompt"
                      type="button" className="flex h-7 w-7 items-center justify-center rounded-md text-ink-500 hover:bg-brand-50 hover:text-brand-600 cursor-pointer">
                      <Icon name={isEditing ? 'check' : 'tune'} className="text-[16px]" />
                    </button>
                    <button onClick={() => handleDeleteClass(c.id)} title="Hapus kelas" aria-label={`Remove class ${c.name || c.id}`}
                      type="button" className="flex h-7 w-7 items-center justify-center rounded-md text-ink-400 hover:bg-danger-50 hover:text-danger-600 cursor-pointer">
                      <Icon name="delete" className="text-[16px]" />
                    </button>
                  </div>
                  {isEditing ? (
                    <textarea
                      aria-label={`Prompt deteksi untuk ${c.name || 'kelas tanpa nama'}`}
                      value={c.prompt}
                      onChange={(e) => setClasses((prev) => prev.map((cls) => cls.id === c.id ? { ...cls, prompt: e.target.value } : cls))}
                      rows={2}
                      className="mt-2 w-full resize-none rounded-md border border-brand-300 p-2 font-mono text-[11px] text-ink-900 focus:outline-none"
                    />
                  ) : (
                    <p className="mt-1.5 line-clamp-2 pl-[18px] text-[11px] text-ink-500">Prompt: “{c.prompt}”</p>
                  )}
                </div>
              );
            })}
          </div>
          {!hasValidClasses && (
            <p id="class-name-error" role="alert" className="mt-2 text-xs text-danger-600">
              {classes.length === 0 ? 'Tambahkan kelas sebelum memberi label.' : 'Isi nama setiap kelas sebelum memberi label.'}
            </p>
          )}
          <Button variant="secondary" size="sm" icon="add" className="mt-3" onClick={handleAddClass}>Tambah kelas</Button>

          <div className="mt-7 space-y-5">
            <div>
              <div className="flex items-center justify-between">
                <label htmlFor="confidence-threshold" className="text-[13px] font-semibold text-ink-900">Ambang confidence</label>
                <span className="text-sm font-semibold tabular-nums text-brand-600">{confidence.toFixed(2)}</span>
              </div>
              <input id="confidence-threshold" type="range" min="0" max="100" value={Math.round(confidence * 100)}
                onChange={(e) => setConfidence(Number(e.target.value) / 100)} className="mt-2 w-full accent-brand-600 cursor-pointer" />
              <div className="flex justify-between text-[10px] text-ink-400"><span>Lebih banyak objek</span><span>Lebih presisi</span></div>
            </div>
            <div>
              <div className="flex items-center justify-between">
                <label htmlFor="iou-threshold" className="text-[13px] font-semibold text-ink-900">Ambang tumpang tindih (IoU)</label>
                <span className="text-sm font-semibold tabular-nums text-brand-600">{iouThreshold.toFixed(2)}</span>
              </div>
              <input id="iou-threshold" type="range" min="0" max="100" value={Math.round(iouThreshold * 100)}
                onChange={(e) => setIouThreshold(Number(e.target.value) / 100)} className="mt-2 w-full accent-brand-600 cursor-pointer" />
            </div>
            <label className="flex items-center justify-between gap-3 rounded-lg bg-brand-50 px-4 py-3 cursor-pointer">
              <span>
                <span className="block text-[13px] font-semibold text-ink-900">Filter ketat model bahasa-visual</span>
                <span className="block text-[11px] text-ink-500">Buang kotak samar yang tidak lolos pengecekan.</span>
              </span>
              <input type="checkbox" checked={strictVlm} onChange={(e) => setStrictVlm(e.target.checked)} className="h-4 w-4 accent-brand-600" />
            </label>
            <div className="flex items-center justify-between rounded-lg bg-brand-50 px-4 py-3">
              <label htmlFor="max-instances" className="text-[13px] font-semibold text-ink-900">Maks. objek per gambar</label>
              <div className="flex items-center gap-1.5">
                <button type="button" aria-label="Kurangi" onClick={() => setMaxInstances((prev) => Math.max(1, prev - 5))}
                  className="flex h-7 w-7 items-center justify-center rounded-md border border-line-strong bg-white text-ink-700 cursor-pointer">−</button>
                <input id="max-instances" type="number" value={maxInstances} onChange={(e) => setMaxInstances(Number(e.target.value))}
                  className="h-7 w-14 rounded-md border border-line-strong bg-white text-center text-xs tabular-nums focus:outline-none" />
                <button type="button" aria-label="Tambah" onClick={() => setMaxInstances((prev) => Math.min(200, prev + 5))}
                  className="flex h-7 w-7 items-center justify-center rounded-md border border-line-strong bg-white text-ink-700 cursor-pointer">+</button>
              </div>
            </div>
          </div>
        </Card>

        <div className="space-y-6">
          <Card className="sticky top-6 p-6">
            <div className="rounded-md bg-[#eef1f6] px-3 py-1.5 text-xs text-ink-700">Ringkasan pipeline</div>
            <h2 className="mt-5 text-[22px] font-bold leading-tight text-ink-900">Jelas sebelum berjalan.</h2>
            <p className="mt-3 text-sm leading-relaxed text-ink-500">
              Batch hanya memproses gambar yang belum punya kotak sama sekali. Gambar yang sudah direview, diberi label otomatis, atau diimpor dari COCO dilewati.
            </p>
            <div className="mt-5">
              {[
                ['Dataset', `${formatCount(totalImages)} gambar`],
                ['Target batch', `${formatCount(batchTarget)} gambar`],
                ['Pipeline', executionPipelineStr],
                ['Status model', isCloud ? 'Cloud' : `${readyModelCount}/${selectedModels.length} siap`],
              ].map(([label, value]) => (
                <div key={label} className="flex items-start justify-between gap-4 py-2.5 text-sm">
                  <span className="shrink-0 text-ink-500">{label}</span>
                  <span className="text-right font-semibold text-ink-900">{value}</span>
                </div>
              ))}
            </div>

            {showJob ? (
              <div className="mt-4 rounded-xl border border-brand-200 bg-brand-50 p-4">
                <div className="flex items-center justify-between text-xs text-ink-500">
                  <span>{isBatchRunning ? 'Anotasi sedang berjalan' : batchProgress?.error ? 'Batch berhenti' : 'Batch selesai'}</span>
                  {isBatchRunning && <Badge tone="warn">Berjalan</Badge>}
                </div>
                <div className="mt-2 text-[34px] font-bold leading-none tabular-nums text-ink-900">{isBatchRunning ? batchPercent : 100}%</div>
                <ProgressBar value={isBatchRunning ? batchPercent : 100} className="mt-3" />
                <div className="mt-2 truncate text-[11px] text-ink-500" title={batchProgress?.current_image}>
                  {isBatchRunning
                    ? `${formatCount(batchProgress?.current || 0)}/${formatCount(batchProgress?.total || batchTarget)} · ${batchProgress?.current_image || 'menyiapkan…'}`
                    : `${formatCount(batchProgress?.succeeded_count ?? batchProgress?.processed_count ?? 0)} gambar diberi label`}
                </div>
                {!isBatchRunning && onNavigate && (
                  <Button size="sm" variant="secondary" block className="mt-3" onClick={() => onNavigate('review')}>Buka antrean review</Button>
                )}
              </div>
            ) : (
              <div className="mt-4 rounded-xl border border-brand-200 bg-brand-50 p-4">
                <div className="text-xs text-ink-500">Rata-rata confidence pratinjau</div>
                <div className="mt-2 text-[28px] font-bold leading-none text-ink-900">
                  {liveMeanConf !== null ? `${Math.round(liveMeanConf * 100)}%` : '—'}
                </div>
                <div className="mt-2 text-[11px] text-ink-500">Dari {sampleImages.length} gambar sampel di bawah.</div>
              </div>
            )}

            <button
              onClick={handleLaunchBatch}
              disabled={!modelsReady || !hasValidClasses || isRescoring || isBatchRunning || batchTarget === 0}
              title={!modelsReady ? modelStatusText : batchTarget === 0 ? 'Semua gambar sudah punya label atau sudah direview' : undefined}
              type="button"
              className="mt-5 flex h-11 w-full items-center justify-center gap-2 rounded-lg bg-brand-600 text-sm font-semibold text-white hover:bg-brand-700 cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Icon name={isBatchRunning ? 'progress_activity' : 'rocket_launch'} className={`text-[18px] ${isBatchRunning ? 'animate-spin' : ''}`} />
              <span>
                {isBatchRunning
                  ? `Memproses (${batchProgress?.current || 0}/${batchProgress?.total || batchTarget})…`
                  : !modelsReady ? (preparationFailed ? 'Model tidak tersedia' : 'Menunggu model…')
                  : batchTarget === 0 ? 'Tidak ada gambar tanpa label'
                  : `Mulai auto-label batch (${formatCount(batchTarget)} gambar)`}
              </span>
            </button>
            <button
              onClick={handleRescore}
              disabled={!modelsReady || !hasValidClasses || isRescoring || isBatchRunning || sampleImages.length === 0}
              title={!modelsReady ? modelStatusText : undefined}
              type="button"
              className="mt-3 flex h-11 w-full items-center justify-center gap-2 rounded-lg border border-line-strong bg-white text-sm font-semibold text-ink-900 hover:bg-brand-50 cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Icon name={isRescoring ? 'progress_activity' : 'visibility'} className={`text-[18px] ${isRescoring ? 'animate-spin' : ''}`} />
              <span>{isRescoring ? 'Menjalankan pratinjau…' : 'Jalankan pratinjau'}</span>
            </button>
            {onNavigate && (
              <Button variant="ghost" block className="mt-2" onClick={() => onNavigate('review')}>Langsung review manual</Button>
            )}
            <p className="mt-4 text-[11px] leading-relaxed text-ink-500">
              Hasil auto-label ditandai “perlu review” dan tidak ikut ekspor yang disetujui sampai kamu memeriksanya.
            </p>
          </Card>
        </div>
      </div>

      <section className="mt-8">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-[15px] font-semibold text-ink-900">Pratinjau pada sampel</h2>
            <p className="mt-0.5 text-xs text-ink-500">Diperbarui otomatis saat pengaturan berubah dan model siap.</p>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex rounded-lg border border-line-strong p-0.5 text-xs" role="group" aria-label="Tampilan pratinjau">
              {([['both', enableSam2Masks ? 'Kotak + mask' : 'Kotak'], ...(enableSam2Masks ? [['masks', 'Mask saja']] : []), ['raw', 'Gambar asli']] as [typeof viewMode, string][]).map(([mode, label]) => (
                <button key={mode} type="button" onClick={() => setViewMode(mode)} aria-pressed={viewMode === mode}
                  className={`rounded-md px-2.5 py-1 font-medium cursor-pointer ${viewMode === mode ? 'bg-brand-600 text-white' : 'text-ink-500 hover:text-ink-900'}`}>
                  {label}
                </button>
              ))}
            </div>
            <Button size="sm" variant="secondary" icon="shuffle" onClick={handleResampleFrames}>Ambil sampel lain</Button>
          </div>
        </div>
        {sampleImages.length === 0 ? (
          <Card className="flex h-48 flex-col items-center justify-center gap-2 border-dashed text-center">
            <Icon name="folder_open" className="text-[28px] text-ink-400" />
            <p className="text-sm text-ink-500">Impor gambar dulu untuk melihat pratinjau deteksi.</p>
          </Card>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {sampleImages.map((img, imgIdx) => {
              const result = previewDetections[img.path || img.filename] || previewDetections[img.filename];
              const dets = result?.detections || [];
              const shown = viewMode === 'raw' ? [] : viewMode === 'masks'
                ? dets.filter((det: any) => det.polygon_normalized?.length).map((det: any) => ({ ...det, norm_left: 0, norm_top: 0, norm_right: 0, norm_bottom: 0 }))
                : dets;
              return (
                <Card key={`${img.path || img.filename}-${imgIdx}`} className="overflow-hidden">
                  <div className="relative bg-canvas">
                    <ImageWithBoxes
                      src={getImageUrl(img.path || img.image_id || img.filename)}
                      alt={img.filename}
                      width={img.width}
                      height={img.height}
                      boxes={shown}
                      showMasks={enableSam2Masks && viewMode !== 'raw'}
                    />
                    {isRescoring && (
                      <div className="absolute inset-0 flex items-center justify-center bg-canvas/50 text-xs font-semibold text-white">
                        <Icon name="progress_activity" className="mr-2 animate-spin text-[18px]" /> Menjalankan model…
                      </div>
                    )}
                  </div>
                  <div className="flex items-center justify-between gap-3 px-4 py-3 text-xs">
                    <span className="min-w-0 truncate font-medium text-ink-900" title={img.filename}>{img.filename}</span>
                    <span className="shrink-0 text-ink-500">
                      {result?.error ? <span className="text-danger-600">Gagal</span>
                        : !modelsReady ? 'Menunggu model'
                        : result ? `${dets.length} objek${result.mean_confidence ? ` · ${Math.round(result.mean_confidence * 100)}%` : ''}${result.elapsed_seconds ? ` · ${Math.round(result.elapsed_seconds * 1000)} ms` : ''}`
                        : '—'}
                    </span>
                  </div>
                </Card>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
};
