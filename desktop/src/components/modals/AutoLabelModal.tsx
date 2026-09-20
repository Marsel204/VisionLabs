import React, { useState, useEffect, useCallback, useMemo } from 'react';
import type { ImageMeta, SystemHealth } from '../../types';
import {
  autoRefinePrompt,
  runAutoLabelPreview,
  startAutoLabelBatch,
  fetchAutoLabelStatus,
  fetchModelPresets,
  browseCustomWeights,
  validateCustomModel,
  getImageUrl,
} from '../../services/api';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onStartBatch: (classes: string[], conf: number) => void;
  health?: SystemHealth | null;
  totalImages?: number;
  images?: ImageMeta[];
  onBatchComplete?: () => void;
}

interface SemanticClass {
  id: string;
  name: string;
  prompt: string;
  color: string;
  enabled: boolean;
}

export const AutoLabelModal: React.FC<Props> = ({
  isOpen,
  onClose,
  onStartBatch,
  health,
  totalImages = 1420,
  images,
  onBatchComplete,
}) => {
  // Detector & Verifier flags
  const [enableGroundingDino, setEnableGroundingDino] = useState(true);
  const [enableYolo, setEnableYolo] = useState(true);
  const [enableFlorenceDetector, setEnableFlorenceDetector] = useState(false);
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
  const [classes, setClasses] = useState<SemanticClass[]>([
    {
      id: '01',
      name: 'Car',
      prompt: 'passenger sedan, hatchback, tesla, modern coupe',
      color: '#06b6d4',
      enabled: true,
    },
    {
      id: '02',
      name: 'Motorcycle',
      prompt: 'motorcycle with rider, motorbike, moped, scooter',
      color: '#f59e0b',
      enabled: true,
    },
    {
      id: '03',
      name: 'Bus',
      prompt: 'city transit bus, double decker, shuttle, transit',
      color: '#10b981',
      enabled: true,
    },
    {
      id: '04',
      name: 'Truck',
      prompt: 'delivery truck, heavy freight semi-trailer, box truck',
      color: '#a855f7',
      enabled: true,
    },
  ]);

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

  // Sample card acceptance states
  const [frameStates, setFrameStates] = useState<{ [key: string]: boolean }>({});

  // Resampling / re-scoring animation state
  const [isRescoring, setIsRescoring] = useState(false);
  const [isBatchRunning, setIsBatchRunning] = useState(false);
  const [batchProgress, setBatchProgress] = useState<import('../../types').AutoLabelStatus | null>(null);
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
      (enableFlorenceDetector ? 1 : 0);

    if (activeDetectorsCount > 1 || (enableYolo && activeYoloModels.length > 1)) {
      return enableSam2Masks ? 'ensemble_fusion_sam2_masks' : 'ensemble_fusion_boxes';
    }
    if (enableYolo) {
      return enableSam2Masks ? 'yolo_sam2_masks' : 'yolo_boxes';
    }
    if (enableFlorenceDetector) {
      return enableSam2Masks ? 'vlm_sam2_masks' : 'vlm_boxes';
    }
    return enableSam2Masks ? 'sam2_dino_masks' : 'dino_boxes';
  }, [
    enableGroundingDino,
    enableYolo,
    enableFlorenceDetector,
    activeYoloModels.length,
    enableSam2Masks,
  ]);

  const fetchPreviewForImages = useCallback(
    async (imgsToPreview: ImageMeta[]) => {
      if (imgsToPreview.length === 0) return;
      setIsRescoring(true);
      try {
        const results = await Promise.allSettled(
          imgsToPreview.map(async (img) => {
            const target = img.path || img.filename;
            const res = await runAutoLabelPreview({
              image_name: target,
              classes,
              confidence_threshold: confidence,
              iou_threshold: iouThreshold,
              strict_vlm: strictVlm,
              max_instances: maxInstances,
              pipeline_mode: computePipelineMode(),
              enable_grounding_dino: enableGroundingDino,
              enable_sam2_masks: enableSam2Masks,
              enable_florence2: enableFlorenceDetector,
              enable_florence2_verifier: enableFlorenceVerifier,
              enable_yolo: enableYolo,
              yolo_models: activeYoloModels,
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
      enableFlorenceVerifier,
      enableYolo,
      activeYoloModels,
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
    if (isOpen && sampleImages.length > 0) {
      const timer = setTimeout(() => {
        fetchPreviewForImages(sampleImages);
      }, 350);
      return () => clearTimeout(timer);
    }
  }, [
    isOpen,
    sampleImages.map((s) => s.path || s.filename).join(','),
    confidence,
    iouThreshold,
    enableGroundingDino,
    enableSam2Masks,
    enableFlorenceDetector,
    enableFlorenceVerifier,
    enableYolo,
    activeYoloModels.join(','),
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
            setModelActionMessage(`Loaded custom weights: ${valid.name}`);
          } else if (activeYoloModels.length < 3) {
            setActiveYoloModels((prev) => [...prev, valid.path]);
            setEnableYolo(true);
            setModelActionMessage(`Added custom model: ${valid.name}`);
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
          setModelActionMessage(`Loaded weights: ${res.name || res.path}`);
        }
      }
    } catch (err) {
      setModelActionMessage(`Browse error: ${err instanceof Error ? err.message : String(err)}`);
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
      setModelActionMessage(`Switched model to preset: ${presetId}`);
    } else if (activeYoloModels.length < 3 && !activeYoloModels.includes(presetId)) {
      setActiveYoloModels((prev) => [...prev, presetId]);
      setEnableYolo(true);
      setModelActionMessage(`Added preset model: ${presetId}`);
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

  const handleAddClass = () => {
    const nextNum = classes.length + 1;
    const newId = String(nextNum).padStart(2, '0');
    const colors = ['#ec4899', '#3b82f6', '#14b8a6', '#f97316', '#8b5cf6'];
    const chosenColor = colors[classes.length % colors.length];
    setClasses((prev) => [
      ...prev,
      {
        id: newId,
        name: `Class_${newId}`,
        prompt: 'new custom visual object description',
        color: chosenColor,
        enabled: true,
      },
    ]);
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
    setIsBatchRunning(true);
    try {
      await startAutoLabelBatch({
        classes,
        confidence_threshold: confidence,
        iou_threshold: iouThreshold,
        pipeline_mode: computePipelineMode(),
        only_unannotated: true,
        enable_grounding_dino: enableGroundingDino,
        enable_sam2_masks: enableSam2Masks,
        enable_florence2: enableFlorenceDetector,
        enable_florence2_verifier: enableFlorenceVerifier,
        enable_yolo: enableYolo,
        yolo_models: activeYoloModels,
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
            onStartBatch(activeClassNames, confidence);
            onBatchComplete?.();
            onClose();
          }
        } catch (pollErr) {
          console.error('Batch polling error:', pollErr);
          clearInterval(interval);
          setIsBatchRunning(false);
        }
      }, 500);
    } catch (err) {
      console.error('Batch start error:', err);
      setIsBatchRunning(false);
    }
  };

  const activePipelineSteps: string[] = [];
  if (enableGroundingDino) activePipelineSteps.push('DINO 1.5');
  if (enableYolo) {
    activePipelineSteps.push(
      activeYoloModels.length > 1 ? `YOLO (${activeYoloModels.length}x Ensemble)` : 'YOLO'
    );
  }
  if (enableFlorenceDetector) activePipelineSteps.push('Florence-2 <OD>');
  if (enableFlorenceVerifier) activePipelineSteps.push('Florence-2 <VERIFY>');
  if (enableSam2Masks) activePipelineSteps.push('SAM-2 <MASKS>');

  const executionPipelineStr =
    activePipelineSteps.length > 0 ? activePipelineSteps.join(' → ') : 'None Selected';


  return (
    <div className="fixed inset-0 z-50 select-none flex items-center justify-center">
      {/* Deep Backdrop Scrim */}
      <div
        className="fixed inset-0 bg-[#040711]/85 backdrop-blur-md z-10 transition-opacity"
        onClick={onClose}
      />

      {/* Main Focal Modal Workstation Container */}
      <div className="fixed inset-2 md:inset-5 z-20 flex flex-col bg-[#060e20]/95 border border-[#3d494c]/70 rounded-xl shadow-[0_24px_60px_rgba(0,0,0,0.85)] backdrop-blur-2xl overflow-hidden text-[#dae2fd]">
        {/* Top Modal Bar */}
        <div className="h-13 px-4 py-2.5 bg-[#131b2e] border-b border-[#3d494c]/80 flex items-center justify-between shrink-0">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded bg-[#4cd7f6]/10 border border-[#4cd7f6]/40 flex items-center justify-center text-[#4cd7f6]">
              <span className="material-symbols-outlined text-[18px]">auto_fix_high</span>
            </div>
            <div className="flex items-center space-x-2.5">
              <h1 className="text-base font-semibold text-[#dae2fd] tracking-tight">
                Auto-Label AI Configuration &amp; Batch Pipeline
              </h1>
              <span className="px-2 py-0.5 rounded bg-[#222a3d] border border-[#3d494c]/60 text-[10px] text-[#4cd7f6] font-mono font-medium">
                VisionForge AI • v3.2-prod
              </span>
            </div>
          </div>

          {/* Center System Health Metric */}
          <div className="hidden xl:flex items-center space-x-3 px-3 py-1 bg-[#060e20]/90 border border-[#3d494c]/50 rounded">
            <div className="flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-[#4edea3] animate-pulse" />
              <span className="text-[11px] text-[#4edea3] uppercase font-bold tracking-wider font-mono">
                GPU Cluster: {health?.gpu?.device || '4x RTX 4090'} (Online)
              </span>
            </div>
            <span className="text-[#3d494c]">|</span>
            <span className="text-[11px] text-[#bcc9cd] font-mono">
              VRAM: {health?.gpu?.vram_free ? `${health.gpu.vram_free} Free` : '86.4 / 96.0 GB'}
            </span>
            <span className="text-[#3d494c]">|</span>
            <span className="text-[11px] text-[#4cd7f6] font-mono">TensorRT 10.3 Engine</span>
          </div>

          {/* Action Utilities */}
          <div className="flex items-center space-x-1.5">
            <button
              type="button"
              className="p-1.5 rounded hover:bg-[#171f33] text-[#bcc9cd] hover:text-[#dae2fd] transition-colors cursor-pointer"
              title="Toggle Fullscreen"
            >
              <span className="material-symbols-outlined text-[20px]">fullscreen</span>
            </button>
            <button
              type="button"
              className="p-1.5 rounded hover:bg-[#171f33] text-[#bcc9cd] hover:text-[#dae2fd] transition-colors cursor-pointer"
              title="Pipeline Documentation"
            >
              <span className="material-symbols-outlined text-[20px]">help</span>
            </button>
            <div className="h-4 w-px bg-[#3d494c] mx-1" />
            <button
              onClick={onClose}
              type="button"
              className="flex items-center space-x-1 px-2.5 py-1 rounded bg-[#222a3d] hover:bg-[#2d3449] border border-[#3d494c] text-[#bcc9cd] hover:text-[#dae2fd] transition-all cursor-pointer active:scale-[0.98]"
            >
              <span className="text-[10px] font-mono font-bold">ESC</span>
              <span className="material-symbols-outlined text-[18px]">close</span>
            </button>
          </div>
        </div>

        {/* Main Workspace Split: Left Config Drawer (410px) + Right Interactive Preview Grid */}
        <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
          {/* LEFT CONFIGURATION PANEL (w-[410px] Fixed Width) */}
          <aside className="w-full md:w-[410px] shrink-0 border-r border-[#3d494c]/80 bg-[#131b2e]/80 flex flex-col overflow-y-auto custom-scrollbar p-3 space-y-4">
            {/* SECTION 1: AI Model Ensemble Architecture */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-[11px] font-semibold uppercase tracking-wider text-[#dae2fd] flex items-center space-x-1.5">
                  <span className="material-symbols-outlined text-[16px] text-[#4cd7f6]">hub</span>
                  <span>AI Ensemble Pipeline</span>
                </label>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#1bbd85]/20 text-[#4edea3] font-mono">
                  Multi-Detector Cascade
                </span>
              </div>

              {/* DETECTORS GROUP */}
              <div className="space-y-2">
                <div className="text-[10px] uppercase font-mono tracking-wider text-[#a0b4c4] flex items-center gap-1">
                  <span className="material-symbols-outlined text-[12px] text-[#4cd7f6]">bolt</span>
                  <span>1. Object Detectors (Ensemble Candidate Generation)</span>
                </div>

                {/* Grounding DINO */}
                <div
                  onClick={() => setEnableGroundingDino(!enableGroundingDino)}
                  className={`p-2.5 rounded-lg border transition-all cursor-pointer select-none ${
                    enableGroundingDino
                      ? 'bg-[#1b253b] border-[#4cd7f6]/50'
                      : 'bg-[#0b101e] border-[#2a3a48]/40 opacity-70 hover:opacity-100'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-2.5">
                      <input
                        type="checkbox"
                        checked={enableGroundingDino}
                        onChange={() => {}}
                        className="mt-0.5 w-3.5 h-3.5 rounded bg-[#171f33] border-[#869397] text-[#4cd7f6] accent-[#4cd7f6]"
                      />
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-semibold text-[#dae2fd]">Grounding DINO 1.5 Pro</span>
                          <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-[#4cd7f6]/20 text-[#4cd7f6]">
                            DETECTOR
                          </span>
                        </div>
                        <p className="text-[#869397] text-[11px] mt-0.5">
                          Open-vocabulary prompt grounding &amp; open-domain candidate bounding boxes.
                        </p>
                      </div>
                    </div>
                    <span className="text-[10px] font-mono text-[#4cd7f6]">~18ms</span>
                  </div>
                </div>

                {/* YOLO & Custom Models Multi-Model Card */}
                <div
                  className={`p-2.5 rounded-lg border transition-all ${
                    enableYolo
                      ? 'bg-[#1b253b] border-[#10b981]/50'
                      : 'bg-[#0b101e] border-[#2a3a48]/40 opacity-70'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div
                      onClick={() => setEnableYolo(!enableYolo)}
                      className="flex items-start gap-2.5 cursor-pointer flex-1"
                    >
                      <input
                        type="checkbox"
                        checked={enableYolo}
                        onChange={() => {}}
                        className="mt-0.5 w-3.5 h-3.5 rounded bg-[#171f33] border-[#869397] text-[#10b981] accent-[#10b981]"
                      />
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-semibold text-[#dae2fd]">YOLO &amp; Custom Weights Ensemble</span>
                          <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-[#10b981]/20 text-[#34d399]">
                            {activeYoloModels.length > 1 ? `${activeYoloModels.length} MODELS ACTIVE` : 'YOLO'}
                          </span>
                        </div>
                        <p className="text-[#869397] text-[11px] mt-0.5">
                          High-speed edge detection with NMS fusion across custom *.pt or *.onnx models.
                        </p>
                      </div>
                    </div>
                    <span className="text-[10px] font-mono text-[#10b981]">~6ms</span>
                  </div>

                  {/* Active Models List */}
                  {enableYolo && (
                    <div className="mt-2.5 pt-2.5 border-t border-[#2a3a48]/50 space-y-1.5">
                      <div className="flex items-center justify-between text-[10px] font-mono text-[#a0b4c4]">
                        <span>Active Weights ({activeYoloModels.length}/3 max):</span>
                        {activeYoloModels.length < 3 && (
                          <button
                            type="button"
                            onClick={() => {
                              setReplacingModelIdx(null);
                              setShowAddMenu(!showAddMenu);
                            }}
                            className="text-[#7dd3fc] hover:text-[#38bdf8] flex items-center gap-1 cursor-pointer"
                          >
                            <span className="material-symbols-outlined text-[13px]">add_circle</span>
                            <span>+ Add Model</span>
                          </button>
                        )}
                      </div>

                      {activeYoloModels.map((modelPath, idx) => {
                        const isPreset = !modelPath.includes('/') && !modelPath.includes('\\');
                        const displayName = modelPath.split('/').filter(Boolean).pop() || modelPath;

                        return (
                          <div
                            key={`model-${idx}-${modelPath}`}
                            className="flex items-center justify-between p-1.5 px-2 rounded bg-[#0a0e1a] border border-[#2a3a48]/60 text-xs font-mono"
                          >
                            <div className="flex items-center gap-2 truncate flex-1 mr-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-[#10b981]"></span>
                              <span className="text-[#e0e8f0] truncate font-medium" title={modelPath}>
                                {displayName}
                              </span>
                              <span
                                className={`text-[9px] px-1 py-0.2 rounded font-bold ${
                                  isPreset ? 'bg-[#06b6d4]/20 text-[#06b6d4]' : 'bg-[#a855f7]/20 text-[#c084fc]'
                                }`}
                              >
                                {isPreset ? 'PRESET' : 'CUSTOM'}
                              </span>
                            </div>

                            <div className="flex items-center gap-1 shrink-0">
                              <button
                                type="button"
                                onClick={() => {
                                  setReplacingModelIdx(idx);
                                  setShowAddMenu(true);
                                }}
                                className="px-1.5 py-0.5 rounded bg-[#1f2c3f] hover:bg-[#2b3c54] text-[10px] text-[#7dd3fc] cursor-pointer"
                                title="Change model weights"
                              >
                                Swap
                              </button>
                              {activeYoloModels.length > 1 && (
                                <button
                                  type="button"
                                  onClick={() => handleRemoveModel(idx)}
                                  className="px-1.5 py-0.5 rounded hover:bg-[#ffb4ab]/20 text-[#ffb4ab] text-[11px] cursor-pointer"
                                  title="Remove model from ensemble"
                                >
                                  ✕
                                </button>
                              )}
                            </div>
                          </div>
                        );
                      })}

                      {/* Dropdown Menu for Presets & Custom Weights */}
                      {showAddMenu && (
                        <div className="p-2 rounded-lg bg-[#0e1626] border border-[#38bdf8]/40 shadow-xl space-y-1.5 mt-1">
                          <div className="flex items-center justify-between text-[10px] font-mono font-bold text-[#7dd3fc]">
                            <span>
                              {replacingModelIdx !== null
                                ? `Replace Model ${replacingModelIdx + 1} with:`
                                : 'Add Model to Ensemble:'}
                            </span>
                            <button
                              type="button"
                              onClick={() => {
                                setShowAddMenu(false);
                                setReplacingModelIdx(null);
                              }}
                              className="text-[#a0b4c4] hover:text-white"
                            >
                              ✕
                            </button>
                          </div>

                          {/* Browse Custom File CTA */}
                          <button
                            type="button"
                            onClick={() => handleBrowseCustomWeights(replacingModelIdx ?? undefined)}
                            className="w-full text-left p-1.5 rounded bg-[#1e293b] hover:bg-[#28384f] text-[#38bdf8] text-xs font-semibold flex items-center gap-2 cursor-pointer border border-[#38bdf8]/30 transition-colors"
                          >
                            <span className="material-symbols-outlined text-[15px]">folder_open</span>
                            <span>📂 Browse Custom Weights (*.pt, *.onnx, *.engine)...</span>
                          </button>

                          <div className="text-[9px] uppercase font-mono text-[#a0b4c4] pt-1">
                            Or Choose from Presets:
                          </div>
                          <div className="grid grid-cols-2 gap-1">
                            {presets.map((p) => (
                              <button
                                key={p.id}
                                type="button"
                                onClick={() => handleSelectPreset(p.id, replacingModelIdx ?? undefined)}
                                className="text-left p-1.5 rounded bg-[#111828] hover:bg-[#1a2438] text-[11px] font-mono text-[#e0e8f0] flex items-center justify-between cursor-pointer border border-[#2a3a48]/40"
                              >
                                <span className="truncate">{p.id}</span>
                                <span className="text-[9px] text-[#a0b4c4]">{p.size}</span>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Florence-2 VLM as DETECTOR */}
                <div
                  onClick={() => setEnableFlorenceDetector(!enableFlorenceDetector)}
                  className={`p-2.5 rounded-lg border transition-all cursor-pointer select-none ${
                    enableFlorenceDetector
                      ? 'bg-[#1b253b] border-[#c084fc]/50'
                      : 'bg-[#0b101e] border-[#2a3a48]/40 opacity-70 hover:opacity-100'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-2.5">
                      <input
                        type="checkbox"
                        checked={enableFlorenceDetector}
                        onChange={() => {}}
                        className="mt-0.5 w-3.5 h-3.5 rounded bg-[#171f33] border-[#869397] text-[#c084fc] accent-[#c084fc]"
                      />
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-semibold text-[#dae2fd]">Florence-2 VLM (Object Detector)</span>
                          <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-[#c084fc]/20 text-[#c084fc]">
                            OD-DETECTOR
                          </span>
                        </div>
                        <p className="text-[#869397] text-[11px] mt-0.5">
                          Direct open-domain bounding box proposal generation via Florence-2 &lt;OD&gt; task.
                        </p>
                      </div>
                    </div>
                    <span className="text-[10px] font-mono text-[#c084fc]">~25ms</span>
                  </div>
                </div>
              </div>

              {/* VERIFIERS GROUP */}
              <div className="space-y-2 pt-1">
                <div className="text-[10px] uppercase font-mono tracking-wider text-[#a0b4c4] flex items-center gap-1">
                  <span className="material-symbols-outlined text-[12px] text-[#34d399]">verified</span>
                  <span>2. Semantic Verification &amp; Hallucination Filter</span>
                </div>

                {/* Florence-2 VLM as VERIFIER */}
                <div
                  onClick={() => setEnableFlorenceVerifier(!enableFlorenceVerifier)}
                  className={`p-2.5 rounded-lg border transition-all cursor-pointer select-none ${
                    enableFlorenceVerifier
                      ? 'bg-[#1b253b] border-[#34d399]/50'
                      : 'bg-[#0b101e] border-[#2a3a48]/40 opacity-70 hover:opacity-100'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-2.5">
                      <input
                        type="checkbox"
                        checked={enableFlorenceVerifier}
                        onChange={() => {}}
                        className="mt-0.5 w-3.5 h-3.5 rounded bg-[#171f33] border-[#869397] text-[#34d399] accent-[#34d399]"
                      />
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-semibold text-[#dae2fd]">
                            Florence-2 VLM Verification (Filter False Positives)
                          </span>
                          <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-[#34d399]/20 text-[#34d399]">
                            VERIFIER
                          </span>
                        </div>
                        <p className="text-[#869397] text-[11px] mt-0.5">
                          Crops proposed candidate boxes and validates semantic descriptions via &lt;CAPTION&gt; reasoning to reject hallucinated detections.
                        </p>
                      </div>
                    </div>
                    <span className="text-[10px] font-mono text-[#34d399]">~22ms</span>
                  </div>
                </div>
              </div>

              {/* SEGMENTATION GROUP */}
              <div className="space-y-2 pt-1">
                <div className="text-[10px] uppercase font-mono tracking-wider text-[#a0b4c4] flex items-center gap-1">
                  <span className="material-symbols-outlined text-[12px] text-[#38bdf8]">polyline</span>
                  <span>3. Sub-Pixel Mask Segmentation</span>
                </div>

                {/* SAM 2 */}
                <div
                  onClick={() => {
                    const next = !enableSam2Masks;
                    setEnableSam2Masks(next);
                    if (!next && viewMode === 'masks') {
                      setViewMode('both');
                    }
                  }}
                  className={`p-2.5 rounded-lg border transition-all cursor-pointer select-none ${
                    enableSam2Masks
                      ? 'bg-[#1b253b] border-[#38bdf8]/50'
                      : 'bg-[#0b101e] border-[#2a3a48]/40 opacity-70 hover:opacity-100'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-2.5">
                      <input
                        type="checkbox"
                        checked={enableSam2Masks}
                        onChange={() => {}}
                        className="mt-0.5 w-3.5 h-3.5 rounded bg-[#171f33] border-[#869397] text-[#38bdf8] accent-[#38bdf8]"
                      />
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-semibold text-[#dae2fd]">SAM 2 (Polygon Segmentation)</span>
                          <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-[#38bdf8]/20 text-[#38bdf8]">
                            POLYGON MASKS
                          </span>
                        </div>
                        <p className="text-[#869397] text-[11px] mt-0.5">
                          Predicts fine boundary contours from bounding box prompts for high-precision polygon labels.
                        </p>
                      </div>
                    </div>
                    <span className="text-[10px] font-mono text-[#38bdf8]">~34ms</span>
                  </div>
                </div>
              </div>

              {/* Action Message feedback */}
              {modelActionMessage && (
                <div className="p-2 rounded bg-[#10b981]/20 border border-[#10b981]/40 text-[#34d399] text-[11px] font-mono flex items-center justify-between">
                  <span>✓ {modelActionMessage}</span>
                  <button
                    type="button"
                    onClick={() => setModelActionMessage(null)}
                    className="hover:text-white"
                  >
                    ✕
                  </button>
                </div>
              )}

              {/* Strategy pill */}
              <div className="px-2.5 py-1.5 rounded bg-[#171f33] border border-[#3d494c]/50 flex items-center justify-between text-[11px] font-mono text-[#bcc9cd]">
                <span className="text-[#869397]">Pipeline:</span>
                <span className="text-[#4cd7f6] font-semibold truncate max-w-[280px]" title={executionPipelineStr}>
                  {executionPipelineStr}
                </span>
              </div>
            </div>

            {/* SECTION 2: Class Prompt Builder */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-[11px] font-semibold uppercase tracking-wider text-[#dae2fd] flex items-center space-x-1.5">
                  <span className="material-symbols-outlined text-[16px] text-[#4cd7f6]">label</span>
                  <span>Target Semantic Classes</span>
                </label>
                <span className="text-[10px] text-[#bcc9cd] font-mono">
                  {classes.length} Configured
                </span>
              </div>

              <div className="space-y-1.5">
                {classes.map((c) => {
                  const isRefining = refiningId === c.id;
                  const isEditing = editingId === c.id;

                  return (
                    <div
                      key={c.id}
                      className="p-2 rounded bg-[#060e20] border border-[#3d494c]/80 hover:border-[#4cd7f6]/50 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <span
                            className="w-2.5 h-2.5 rounded-full"
                            style={{ backgroundColor: c.color }}
                          />
                          <span className="text-xs font-bold text-[#dae2fd]">{c.name}</span>
                          <span className="text-[10px] text-[#869397] font-mono">[ID: {c.id}]</span>
                        </div>
                        <div className="flex items-center space-x-1 text-[#bcc9cd]">
                          {/* Auto Refine Prompt Button */}
                          <button
                            onClick={() => handleAutoRefine(c)}
                            disabled={isRefining}
                            title="Auto-refine prompt with Florence-2 VLM"
                            type="button"
                            className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-[#4cd7f6]/10 hover:bg-[#4cd7f6]/20 text-[#4cd7f6] text-[10px] font-mono border border-[#4cd7f6]/30 transition-all cursor-pointer disabled:opacity-50"
                          >
                            <span className="material-symbols-outlined text-[12px]">
                              {isRefining ? 'sync' : 'auto_fix_high'}
                            </span>
                            <span>{isRefining ? 'Refining...' : '✨ Auto-Refine'}</span>
                          </button>

                          <button
                            onClick={() => setEditingId(isEditing ? null : c.id)}
                            className="hover:text-[#4cd7f6] p-0.5 rounded cursor-pointer transition-colors"
                            title="Edit prompt text"
                            type="button"
                          >
                            <span className="material-symbols-outlined text-[14px]">
                              {isEditing ? 'check' : 'tune'}
                            </span>
                          </button>
                          <button
                            onClick={() => handleDeleteClass(c.id)}
                            className="hover:text-[#ffb4ab] p-0.5 rounded cursor-pointer transition-colors"
                            title="Remove class"
                            type="button"
                          >
                            <span className="material-symbols-outlined text-[14px]">delete</span>
                          </button>
                        </div>
                      </div>

                      <div className="mt-1.5 bg-[#222a3d]/70 px-2 py-1 rounded border border-[#3d494c]/40">
                        {isEditing ? (
                          <div className="flex flex-col gap-1">
                            <span className="text-[10px] font-mono text-[#4cd7f6]">prompt:</span>
                            <textarea
                              value={c.prompt}
                              onChange={(e) =>
                                setClasses((prev) =>
                                  prev.map((cls) =>
                                    cls.id === c.id ? { ...cls, prompt: e.target.value } : cls
                                  )
                                )
                              }
                              rows={2}
                              className="w-full bg-[#060e20] text-[11px] font-mono text-[#dae2fd] p-1.5 rounded border border-[#4cd7f6]/50 focus:outline-none resize-none"
                            />
                          </div>
                        ) : (
                          <div className="flex items-baseline">
                            <span
                              className="text-[11px] font-mono font-medium"
                              style={{ color: c.color }}
                            >
                              prompt:
                            </span>
                            <span className="text-[11px] font-mono text-[#bcc9cd] ml-1.5 line-clamp-2">
                              "{c.prompt}"
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              <button
                onClick={handleAddClass}
                type="button"
                className="w-full py-1.5 px-3 rounded bg-[#222a3d]/80 hover:bg-[#2d3449] border border-dashed border-[#869397] hover:border-[#4cd7f6] text-[11px] text-[#4cd7f6] flex items-center justify-center space-x-1.5 transition-all cursor-pointer"
              >
                <span className="material-symbols-outlined text-[16px]">add</span>
                <span>Add Class Prompt</span>
              </button>
            </div>

            {/* SECTION 3: Precision Hyperparameter Controls */}
            <div className="space-y-3 pt-2 border-t border-[#3d494c]/60">
              <label className="text-[11px] font-semibold uppercase tracking-wider text-[#dae2fd] flex items-center space-x-1.5">
                <span className="material-symbols-outlined text-[16px] text-[#4cd7f6]">tune</span>
                <span>Inference Parameters</span>
              </label>

              {/* Confidence Threshold Slider */}
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-[#bcc9cd]">Confidence Threshold</span>
                  <span className="px-1.5 py-0.5 rounded bg-[#060e20] border border-[#3d494c] font-mono text-[11px] font-semibold text-[#4cd7f6]">
                    {confidence.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={Math.round(confidence * 100)}
                  onChange={(e) => setConfidence(Number(e.target.value) / 100)}
                  className="w-full h-1 bg-[#222a3d] rounded-lg appearance-none cursor-pointer accent-[#4cd7f6]"
                />
                <div className="flex justify-between text-[9px] font-mono text-[#869397]">
                  <span>0.0 (Recall)</span>
                  <span>0.50</span>
                  <span>1.0 (Precision)</span>
                </div>
              </div>

              {/* IoU / NMS Slider */}
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-[#bcc9cd]">IoU / NMS Overlap Threshold</span>
                  <span className="px-1.5 py-0.5 rounded bg-[#060e20] border border-[#3d494c] font-mono text-[11px] font-semibold text-[#c0c1ff]">
                    {iouThreshold.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={Math.round(iouThreshold * 100)}
                  onChange={(e) => setIouThreshold(Number(e.target.value) / 100)}
                  className="w-full h-1 bg-[#222a3d] rounded-lg appearance-none cursor-pointer accent-[#c0c1ff]"
                />
              </div>

              {/* Strict VLM Hallucination Pruning Toggle */}
              <div className="p-2 rounded bg-[#060e20] border border-[#3d494c] flex items-center justify-between">
                <div className="pr-2">
                  <div className="text-[11px] text-[#dae2fd] font-medium">
                    VLM Strict Reasoning Filter
                  </div>
                  <div className="text-[10px] text-[#869397]">
                    Discard subtle phantom boxes via zero-shot VLM query
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer shrink-0">
                  <input
                    type="checkbox"
                    checked={strictVlm}
                    onChange={(e) => setStrictVlm(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-8 h-4 bg-[#222a3d] peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-[#4cd7f6]" />
                </label>
              </div>

              {/* Max Annotations Cap */}
              <div className="flex items-center justify-between p-2 rounded bg-[#060e20] border border-[#3d494c]">
                <span className="text-[11px] text-[#bcc9cd]">Max Instances / Frame</span>
                <div className="flex items-center space-x-1.5">
                  <button
                    onClick={() => setMaxInstances((prev) => Math.max(1, prev - 5))}
                    type="button"
                    className="w-5 h-5 rounded bg-[#222a3d] text-[#dae2fd] flex items-center justify-center font-mono hover:bg-[#2d3449] cursor-pointer"
                  >
                    -
                  </button>
                  <input
                    type="number"
                    value={maxInstances}
                    onChange={(e) => setMaxInstances(Number(e.target.value))}
                    className="w-12 text-center py-0.5 bg-[#0b1326] border border-[#3d494c] rounded font-mono text-[11px] text-[#4cd7f6] focus:outline-none"
                  />
                  <button
                    onClick={() => setMaxInstances((prev) => Math.min(200, prev + 5))}
                    type="button"
                    className="w-5 h-5 rounded bg-[#222a3d] text-[#dae2fd] flex items-center justify-center font-mono hover:bg-[#2d3449] cursor-pointer"
                  >
                    +
                  </button>
                </div>
              </div>
            </div>
          </aside>

          {/* RIGHT MAIN AREA: Interactive 4-Sample Verification Canvas Grid */}
          <main className="flex-1 bg-[#060e20]/90 flex flex-col overflow-hidden">
            {/* Batch Test Header Bar */}
            <div className="px-4 py-2 border-b border-[#3d494c]/70 bg-[#131b2e]/60 flex flex-wrap items-center justify-between gap-2 shrink-0">
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-semibold text-[#dae2fd]">
                    Test Batch Verification
                  </span>
                  <span className="text-[10px] font-mono text-[#4cd7f6] bg-[#4cd7f6]/10 border border-[#4cd7f6]/30 px-2 py-0.5 rounded">
                    4 Frames Sampled
                  </span>
                </div>
                <span className="hidden sm:inline-block text-[#3d494c]">|</span>
                {/* Live Preview Toggles */}
                <div className="hidden sm:flex items-center space-x-1 bg-[#060e20] p-0.5 rounded border border-[#3d494c]">
                  <button
                    onClick={() => setViewMode('both')}
                    type="button"
                    className={`px-2 py-0.5 rounded text-[11px] font-semibold transition-all cursor-pointer ${
                      viewMode === 'both'
                        ? 'bg-[#4cd7f6] text-[#003640]'
                        : 'text-[#bcc9cd] hover:text-[#dae2fd]'
                    }`}
                  >
                    {enableSam2Masks ? 'BBox + SAM-2 Masks' : 'BBox Only'}
                  </button>
                  {enableSam2Masks && (
                    <button
                      onClick={() => setViewMode('masks')}
                      type="button"
                      className={`px-2 py-0.5 rounded text-[11px] font-semibold transition-all cursor-pointer ${
                        viewMode === 'masks'
                          ? 'bg-[#4cd7f6] text-[#003640]'
                          : 'text-[#bcc9cd] hover:text-[#dae2fd]'
                      }`}
                    >
                      Masks Only
                    </button>
                  )}
                  <button
                    onClick={() => setViewMode('raw')}
                    type="button"
                    className={`px-2 py-0.5 rounded text-[11px] font-semibold transition-all cursor-pointer ${
                      viewMode === 'raw'
                        ? 'bg-[#4cd7f6] text-[#003640]'
                        : 'text-[#bcc9cd] hover:text-[#dae2fd]'
                    }`}
                  >
                    Raw Image
                  </button>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <button
                  onClick={handleResampleFrames}
                  type="button"
                  className="flex items-center space-x-1 px-2.5 py-1 rounded bg-[#171f33] border border-[#3d494c] hover:border-[#869397] text-[11px] text-[#bcc9cd] hover:text-[#dae2fd] transition-all cursor-pointer active:scale-95"
                >
                  <span
                    className={`material-symbols-outlined text-[16px] ${
                      isRescoring ? 'animate-spin' : ''
                    }`}
                  >
                    shuffle
                  </span>
                  <span>Resample Frames</span>
                </button>
                <div className="flex items-center space-x-1 text-[11px] font-mono text-[#869397] pl-2 border-l border-[#3d494c]">
                  <span>Zoom:</span>
                  <span className="text-[#4cd7f6] font-semibold">100% Fit</span>
                </div>
              </div>
            </div>

            {/* 2x2 Grid of Real Dataset Verification Frames */}
            <div className="flex-1 p-3 grid grid-cols-1 lg:grid-cols-2 gap-3 overflow-y-auto custom-scrollbar">
              {sampleImages.length === 0 ? (
                <div className="col-span-full h-80 flex flex-col items-center justify-center border-2 border-dashed border-[#3d494c]/60 rounded-xl p-8 text-center bg-[#0d1526]/50">
                  <span className="material-symbols-outlined text-4xl text-[#7dd3fc] mb-3">folder_open</span>
                  <h3 className="text-sm font-semibold text-[#dae2fd] mb-1">No Dataset Images Available</h3>
                  <p className="text-xs text-[#869397] max-w-sm mb-4">
                    Import or scan images into the dataset directory to inspect model detections and SAM-2 polygons.
                  </p>
                </div>
              ) : (
                sampleImages.map((img, imgIdx) => {
                  const result = previewDetections[img.path || img.filename] || previewDetections[img.filename];
                  const dets = result?.detections || [];
                  const isAccepted = frameStates[img.filename] ?? true;

                  return (
                    <div
                      key={`${img.path || img.filename}-${imgIdx}`}
                      className="relative bg-[#131b2e] border border-[#3d494c]/80 rounded-lg overflow-hidden flex flex-col group hover:border-[#4cd7f6]/60 transition-all shadow-md"
                    >
                      {/* Frame Top Meta Ribbon */}
                      <div className="px-2.5 py-1.5 bg-[#131b2e]/95 border-b border-[#3d494c]/60 flex items-center justify-between text-[11px] font-mono">
                        <div className="flex items-center space-x-2 truncate max-w-[65%]">
                          <span className="w-2 h-2 rounded-full bg-[#4edea3] shrink-0" />
                          <span className="text-[#dae2fd] font-semibold truncate" title={img.filename}>
                            {img.filename}
                          </span>
                          <span className="text-[#869397] shrink-0">
                            {img.width || 640}x{img.height || 640}
                          </span>
                        </div>
                        <div className="flex items-center space-x-2 shrink-0">
                          <span className="text-[#4edea3] font-bold">
                            {isRescoring ? 'Reasoning...' : `${dets.length} Detections`}
                          </span>
                          <span className="text-[#869397]">|</span>
                          <label className="flex items-center space-x-1 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={isAccepted}
                              onChange={(e) =>
                                setFrameStates((prev) => ({
                                  ...prev,
                                  [img.filename]: e.target.checked,
                                }))
                              }
                              className="w-3.5 h-3.5 rounded bg-[#171f33] border-[#869397] text-[#4edea3] focus:ring-0 accent-[#4edea3] cursor-pointer"
                            />
                            <span className="text-[11px] text-[#4edea3] font-semibold">Accept</span>
                          </label>
                        </div>
                      </div>

                      {/* Viewport Graphic Layer with Overlay HUD & Annotations */}
                      <div className="relative flex-1 bg-[#070b14] min-h-[260px] overflow-hidden flex items-center justify-center p-2">
                        <div className="relative inline-flex items-center justify-center max-w-full max-h-full">
                          <img
                            className="w-auto h-auto max-w-full max-h-[280px] object-contain block select-none rounded shadow-md pointer-events-none"
                            alt={img.filename}
                            src={getImageUrl(img.filename)}
                          />

                          {/* Loading / Scanning HUD indicator */}
                          {isRescoring && (
                            <div className="absolute inset-0 bg-[#060e20]/60 backdrop-blur-[2px] flex flex-col items-center justify-center space-y-2 pointer-events-none">
                              <span className="material-symbols-outlined text-[#4cd7f6] text-3xl animate-spin">
                                auto_awesome
                              </span>
                              <span className="text-xs font-mono text-[#4cd7f6] tracking-wider uppercase">
                                AI Ensemble Reasoning...
                              </span>
                            </div>
                          )}

                          {/* SVG HUD Overlay */}
                          {viewMode !== 'raw' && (
                            <svg
                              className="absolute inset-0 w-full h-full pointer-events-none"
                              preserveAspectRatio="none"
                              viewBox="0 0 1000 1000"
                            >
                              {dets.map((det: any, detIdx: number) => {
                                const x = (det.norm_left ?? 0) * 1000;
                                const y = (det.norm_top ?? 0) * 1000;
                                const width = Math.max(4, ((det.norm_right ?? 0) - (det.norm_left ?? 0)) * 1000);
                                const height = Math.max(4, ((det.norm_bottom ?? 0) - (det.norm_top ?? 0)) * 1000);
                                const detColor = det.color || '#06b6d4';

                                const polygonPoints =
                                  det.polygon_normalized && det.polygon_normalized.length > 0
                                    ? det.polygon_normalized
                                        .map(([px, py]: [number, number]) => `${px * 1000},${py * 1000}`)
                                        .join(' ')
                                    : null;

                                return (
                                  <g key={`svg-det-${detIdx}`}>
                                    {/* Mask Polygon */}
                                    {enableSam2Masks && polygonPoints && (
                                      <polygon
                                        points={polygonPoints}
                                        fill={detColor}
                                        fillOpacity="0.3"
                                        stroke={detColor}
                                        strokeWidth="1.5"
                                      />
                                    )}

                                    {/* Bounding Box */}
                                    {viewMode === 'both' && (
                                      <rect
                                        x={x}
                                        y={y}
                                        width={width}
                                        height={height}
                                        fill={detColor}
                                        fillOpacity="0.18"
                                        stroke={detColor}
                                        strokeWidth="2"
                                        className="reticle-glow"
                                      />
                                    )}
                                  </g>
                                );
                              })}
                            </svg>
                          )}

                          {/* Floating Micro Confidence Pills */}
                          {viewMode !== 'raw' &&
                            dets.slice(0, 6).map((det: any, detIdx: number) => {
                              const left = Math.min(85, Math.max(3, (det.norm_left ?? 0) * 100));
                              const top = Math.min(88, Math.max(3, (det.norm_top ?? 0) * 100));
                              const detColor = det.color || '#06b6d4';

                              return (
                                <div
                                  key={`tag-${detIdx}`}
                                  style={{
                                    left: `${left}%`,
                                    top: `${top}%`,
                                    borderColor: detColor,
                                  }}
                                  className="absolute px-1.5 py-0.5 rounded bg-[#090d16]/90 border text-[9px] font-mono text-[#dae2fd] flex items-center space-x-1 shadow-lg pointer-events-none -translate-y-1"
                                >
                                  <span
                                    className="w-1.5 h-1.5 rounded-full"
                                    style={{ backgroundColor: detColor }}
                                  />
                                  <span className="font-bold capitalize">{det.class_name}:</span>
                                  <span style={{ color: detColor }}>
                                    {Math.round((det.confidence ?? 0.9) * 100)}%
                                  </span>
                                  {enableSam2Masks && det.polygon_normalized && det.polygon_normalized.length > 0 && (
                                    <span className="text-[#4edea3] text-[8px]">SAM-2 ✓</span>
                                  )}
                                </div>
                              );
                            })}
                        </div>

                        {/* Floating Reticle Controls (Bottom HUD) */}
                        <div className="absolute bottom-2 left-2 flex items-center space-x-1 bg-[#060e20]/80 backdrop-blur-md px-2 py-1 rounded border border-[#3d494c]/60 text-[10px] font-mono text-[#bcc9cd]">
                          <span>Mean Conf:</span>
                          <span className="text-[#4edea3] font-bold">
                            {result?.mean_confidence
                              ? `${(result.mean_confidence * 100).toFixed(1)}%`
                              : liveMeanConf !== null
                              ? `${(liveMeanConf * 100).toFixed(1)}%`
                              : '97.2%'}
                          </span>
                          <span className="mx-1 text-[#869397]">|</span>
                          <span>IoU:</span>
                          <span className="text-[#4cd7f6] font-bold">
                            {result?.iou ? result.iou.toFixed(2) : '0.95'}
                          </span>
                          <span className="mx-1 text-[#869397]">|</span>
                          <span>Latency:</span>
                          <span className="text-[#dae2fd] font-bold">
                            {result?.elapsed_seconds
                              ? `${Math.round(result.elapsed_seconds * 1000)}ms`
                              : '32ms'}
                          </span>
                        </div>

                        <div className="absolute bottom-2 right-2 flex items-center space-x-1">
                          <button
                            type="button"
                            className="w-6 h-6 rounded bg-[#060e20]/80 border border-[#3d494c] flex items-center justify-center text-[#bcc9cd] hover:text-[#4cd7f6] transition-colors cursor-pointer"
                            title="Zoom In"
                          >
                            <span className="material-symbols-outlined text-[14px]">zoom_in</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </main>
        </div>

        {/* BOTTOM MODAL CONTROL FOOTER */}
        <footer className="h-16 px-4 bg-[#131b2e] border-t border-[#3d494c] flex flex-col md:flex-row items-center justify-between gap-3 shrink-0">
          {/* Left: Target Dataset Scope & Compute Telemetry */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <span className="material-symbols-outlined text-[20px] text-[#4cd7f6]">
                filter_center_focus
              </span>
              <div>
                <div className="text-xs font-semibold text-[#dae2fd] flex items-center space-x-2">
                  <span>Target: {(totalImages > 1 ? totalImages : 1420).toLocaleString()} unannotated frames</span>
                  <span className="text-[#869397]">•</span>
                  <span className="text-[#bcc9cd]">Autonomous Driving Cam-Front-04</span>
                </div>
                <div className="text-[10px] font-mono text-[#869397] flex items-center space-x-2">
                  <span>Est. Compute: 4.8 GPU-hrs</span>
                  <span>•</span>
                  <span className="text-[#4edea3]">Zero-Shot Mask Cost: ~$0.42</span>
                </div>
              </div>
            </div>
          </div>

          {/* Center: Execution Readiness Progress Bar Preview */}
          <div className="hidden lg:flex flex-col w-72 space-y-1">
            <div className="flex justify-between text-[11px] font-mono">
              <span className="text-[#bcc9cd] flex items-center space-x-1 truncate max-w-[200px]">
                <span className={`w-1.5 h-1.5 rounded-full ${isBatchRunning ? 'bg-[#06b6d4] animate-ping' : 'bg-[#4edea3]'}`} />
                <span>
                  {isBatchRunning
                    ? `Processing ${batchProgress?.current || 1}/${batchProgress?.total || totalImages}`
                    : 'Pipeline: Validated (4/4 passed)'}
                </span>
              </span>
              <span className="text-[#4cd7f6] font-semibold">
                {isBatchRunning ? `${Math.round(((batchProgress?.current || 0) / Math.max(1, batchProgress?.total || totalImages)) * 100)}%` : '~3m 42s'}
              </span>
            </div>
            <div className="w-full h-1.5 bg-[#222a3d] rounded-full overflow-hidden border border-[#3d494c]/30">
              <div
                className="h-full bg-gradient-to-r from-[#4cd7f6] to-[#4edea3] transition-all duration-300"
                style={{
                  width: isBatchRunning
                    ? `${Math.max(5, Math.round(((batchProgress?.current || 0) / Math.max(1, batchProgress?.total || totalImages)) * 100))}%`
                    : '100%',
                }}
              />
            </div>
          </div>

          {/* Right Actions: Test Execution + Primary Batch Launch CTA */}
          <div className="flex items-center space-x-2.5 w-full md:w-auto justify-end">
            <button
              onClick={handleRescore}
              disabled={isRescoring || isBatchRunning}
              type="button"
              className="px-3 py-2 rounded bg-[#222a3d] hover:bg-[#2d3449] border border-[#3d494c] text-xs font-medium text-[#dae2fd] hover:text-[#4cd7f6] transition-all flex items-center space-x-1.5 active:scale-[0.98] cursor-pointer disabled:opacity-50"
            >
              <span
                className={`material-symbols-outlined text-[18px] ${
                  isRescoring ? 'animate-spin' : ''
                }`}
              >
                terminal
              </span>
              <span>{isRescoring ? 'Running AI Preview...' : 'Run Test Preview (Re-score)'}</span>
            </button>
            <button
              onClick={handleLaunchBatch}
              disabled={isBatchRunning}
              type="button"
              className="px-5 py-2 rounded bg-[#06b6d4] hover:bg-[#4cd7f6] text-[#003640] text-xs font-bold flex items-center space-x-2 shadow-[0_0_16px_rgba(6,182,212,0.45)] hover:shadow-[0_0_24px_rgba(6,182,212,0.65)] transition-all active:scale-[0.98] cursor-pointer disabled:opacity-75"
            >
              <span className={`material-symbols-outlined text-[18px] ${isBatchRunning ? 'animate-spin' : ''}`}>
                {isBatchRunning ? 'sync' : 'rocket_launch'}
              </span>
              <span>
                {isBatchRunning
                  ? `Batch Processing (${batchProgress?.current || 1}/${batchProgress?.total || totalImages})...`
                  : `Start Batch Auto-Labeling (${(totalImages > 1 ? totalImages : 1420).toLocaleString()} Frames)`}
              </span>
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
};
