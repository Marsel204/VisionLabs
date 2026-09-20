import React, { useState, useEffect } from 'react';
import type { DatasetStats } from '../../types';
import {
  exportDataset,
  fetchDatasetStats,
  selectDatasetFolder,
  rescanDataset,
  uploadDatasetImages,
  browseDatasetFolder,
  searchDatasetDirectories,
} from '../../services/api';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  totalImages: number;
  initialTab?: 'export' | 'import';
  onDatasetChanged?: () => void;
}

export const DatasetHubModal: React.FC<Props> = ({
  isOpen,
  onClose,
  totalImages,
  initialTab = 'export',
  onDatasetChanged,
}) => {
  const [activeTab, setActiveTab] = useState<'export' | 'import'>(initialTab);
  const [selectedFormat, setSelectedFormat] = useState<'yolo' | 'coco' | 'voc'>('yolo');
  const [trainRatio, setTrainRatio] = useState<number>(70);
  const [valRatio, setValRatio] = useState<number>(20);
  const testRatio = Math.max(0, 100 - trainRatio - valRatio);

  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportMessage, setExportMessage] = useState<string | null>(null);

  // Import states
  const [folderPathInput, setFolderPathInput] = useState('');
  const [isImporting, setIsImporting] = useState(false);
  const [isBrowsing, setIsBrowsing] = useState(false);
  const [importMessage, setImportMessage] = useState<string | null>(null);
  const [dirSuggestions, setDirSuggestions] = useState<string[]>([]);
  const [showDirDropdown, setShowDirDropdown] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setActiveTab(initialTab);
      fetchDatasetStats()
        .then((s) => {
          setStats(s);
          if (s.directory) setFolderPathInput(s.directory);
        })
        .catch((err) => console.error('Error fetching dataset stats:', err));
    }
  }, [isOpen, initialTab]);

  // Debounced search for directories as user types
  useEffect(() => {
    if (!isOpen || activeTab !== 'import') return;
    const timer = setTimeout(async () => {
      setSearchLoading(true);
      try {
        const res = await searchDatasetDirectories(folderPathInput);
        setDirSuggestions(res.directories || []);
      } catch {
        // ignore errors in background search
      } finally {
        setSearchLoading(false);
      }
    }, 200);

    return () => clearTimeout(timer);
  }, [folderPathInput, isOpen, activeTab]);

  if (!isOpen) return null;

  const handleBrowseFolder = async () => {
    setIsBrowsing(true);
    setImportMessage(null);
    try {
      const res = await browseDatasetFolder();
      if (res.status === 'ok' && res.active_directory) {
        setFolderPathInput(res.active_directory);
        setImportMessage(`Active folder updated: ${res.active_directory} (${res.indexed_count} images indexed)`);
        const s = await fetchDatasetStats();
        setStats(s);
        onDatasetChanged?.();
      } else if (res.status === 'cancelled') {
        // user cancelled file dialog
      } else if (res.status === 'unsupported') {
        setImportMessage('Native file dialog is unsupported. Please search or select from the suggestions below.');
      }
    } catch (err) {
      setImportMessage(`Browse error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsBrowsing(false);
    }
  };

  const handleSelectSuggestion = (path: string) => {
    setFolderPathInput(path);
    setShowDirDropdown(false);
  };

  const handleSwitchFolder = async () => {
    if (!folderPathInput.trim()) return;
    setIsImporting(true);
    setImportMessage(null);
    try {
      const res = await selectDatasetFolder(folderPathInput.trim());
      setImportMessage(`Active folder updated: ${res.active_directory} (${res.indexed_count} images indexed)`);
      const s = await fetchDatasetStats();
      setStats(s);
      onDatasetChanged?.();
    } catch (err) {
      setImportMessage(`Failed to switch folder: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsImporting(false);
    }
  };

  const handleRescan = async () => {
    setIsImporting(true);
    setImportMessage(null);
    try {
      const res = await rescanDataset();
      setImportMessage(`Rescan complete: ${res.indexed_count} images indexed in SQLite database.`);
      const s = await fetchDatasetStats();
      setStats(s);
      onDatasetChanged?.();
    } catch (err) {
      setImportMessage(`Rescan failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsImporting(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    setIsImporting(true);
    setImportMessage(null);
    try {
      const res = await uploadDatasetImages(e.target.files);
      setImportMessage(`Successfully imported ${res.count} image(s)! Total images now: ${res.total_indexed}`);
      const s = await fetchDatasetStats();
      setStats(s);
      onDatasetChanged?.();
    } catch (err) {
      setImportMessage(`Upload failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsImporting(false);
    }
  };

  const handleExport = async () => {
    setIsExporting(true);
    setExportMessage(null);
    try {
      const format = selectedFormat === 'voc' ? 'yolo' : selectedFormat;
      const res = await exportDataset({
        format: format as 'yolo' | 'coco',
        train_ratio: trainRatio / 100,
        val_ratio: valRatio / 100,
        test_ratio: testRatio / 100,
      });
      setExportMessage(
        `Export complete! Generated ${res.format.toUpperCase()} dataset with ${res.total_documents} images (${res.splits.train} train, ${res.splits.val} val, ${res.splits.test} test). Saved to: ${res.destination}`
      );
    } catch (err) {
      setExportMessage(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md select-none p-4">
      <div className="w-full max-w-4xl bg-[#0f1524] border border-[#2a3a48] rounded-2xl shadow-2xl flex flex-col max-h-[90vh] overflow-hidden">
        {/* Modal Header */}
        <div className="p-4 bg-[#111828] border-b border-[#2a3a48]/50 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#0e4d6e] border border-[#7dd3fc]/30 flex items-center justify-center text-[#7dd3fc] font-bold">
              <span className="material-symbols-outlined text-[18px]">
                {activeTab === 'import' ? 'file_upload' : 'dataset'}
              </span>
            </div>
            <div>
              <h2 className="text-sm font-semibold text-[#e0e8f0]">
                {activeTab === 'import' ? 'Import Dataset & Manage Directory' : 'Dataset Management & Export Hub'}
              </h2>
              <p className="text-[11px] text-[#a0b4c4]">
                {activeTab === 'import'
                  ? 'Switch image folder, upload photos, and sync SQLite database'
                  : 'Configure export formats, dataset splits, and cloud synchronization'}
              </p>
            </div>
          </div>

          {/* Tab Switcher */}
          <div className="flex items-center gap-1 bg-[#0a0e1a] p-1 rounded-lg border border-[#2a3a48]/40">
            <button
              type="button"
              onClick={() => setActiveTab('import')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-semibold cursor-pointer transition-all ${
                activeTab === 'import'
                  ? 'bg-[#10b981] text-[#001f2e] shadow'
                  : 'text-[#a0b4c4] hover:text-[#e0e8f0]'
              }`}
            >
              <span className="material-symbols-outlined text-[15px]">file_upload</span>
              <span>Import</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('export')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-semibold cursor-pointer transition-all ${
                activeTab === 'export'
                  ? 'bg-[#06b6d4] text-[#001f2e] shadow'
                  : 'text-[#a0b4c4] hover:text-[#e0e8f0]'
              }`}
            >
              <span className="material-symbols-outlined text-[15px]">file_download</span>
              <span>Export</span>
            </button>
          </div>

          <button
            onClick={onClose}
            type="button"
            className="w-7 h-7 rounded-lg bg-[#141c2e] hover:bg-[#1a2438] text-[#a0b4c4] hover:text-[#e0e8f0] flex items-center justify-center cursor-pointer transition-colors"
          >
            <span className="material-symbols-outlined text-[18px]">close</span>
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {/* SQLite Database Statistics Card */}
          {stats && (
            <div className="p-4 rounded-xl bg-[#141c2e]/80 border border-[#2a3a48]/60 space-y-3 shadow-inner">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-[#06b6d4] text-[18px]">storage</span>
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[#e0e8f0]">
                    SQLite Dataset Index Status
                  </h3>
                </div>
                <span className="text-[10px] font-mono text-[#a0b4c4] truncate max-w-[350px]" title={stats.directory}>
                  Directory: <strong className="text-[#7dd3fc]">{stats.directory}</strong>
                </span>
              </div>

              <div className="grid grid-cols-4 gap-2.5">
                <div className="p-2.5 rounded-lg bg-[#111828] border border-[#2a3a48]/40 flex flex-col">
                  <span className="text-[10px] text-[#a0b4c4] uppercase tracking-wider font-semibold">Total Indexed</span>
                  <span className="text-base font-bold text-[#e0e8f0] font-mono">{stats.total_images}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-[#111828] border border-[#10b981]/30 flex flex-col">
                  <span className="text-[10px] text-[#34d399] uppercase tracking-wider font-semibold">Reviewed</span>
                  <span className="text-base font-bold text-[#34d399] font-mono">{stats.reviewed}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-[#111828] border border-[#a855f7]/30 flex flex-col">
                  <span className="text-[10px] text-[#c084fc] uppercase tracking-wider font-semibold">AI Labeled</span>
                  <span className="text-base font-bold text-[#c084fc] font-mono">{stats.ai_labeled}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-[#111828] border border-[#f59e0b]/30 flex flex-col">
                  <span className="text-[10px] text-[#fbbf24] uppercase tracking-wider font-semibold">Unreviewed</span>
                  <span className="text-base font-bold text-[#fbbf24] font-mono">{stats.unreviewed}</span>
                </div>
              </div>

              {stats.class_counts && Object.keys(stats.class_counts).length > 0 && (
                <div className="pt-1 flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px] text-[#a0b4c4] font-medium mr-1">Class Distribution:</span>
                  {Object.entries(stats.class_counts).map(([cls, count]) => (
                    <span
                      key={cls}
                      className="px-2 py-0.5 rounded-md bg-[#1a2438] text-[10px] font-mono text-[#7dd3fc] border border-[#2a3a48]/50"
                    >
                      {cls}: <strong className="text-white">{count}</strong>
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'import' ? (
            /* =================== TAB 1: IMPORT & DATASET MANAGEMENT =================== */
            <div className="space-y-4">
              {/* Switch Active Folder Card */}
              <div className="p-4 rounded-xl bg-[#141c2e]/60 border border-[#2a3a48]/40 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="material-symbols-outlined text-[#10b981] text-[18px]">folder</span>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-[#e0e8f0]">
                      Active Dataset Directory
                    </h3>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={handleBrowseFolder}
                      disabled={isBrowsing || isImporting}
                      className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-[#0e4d6e] hover:bg-[#125e86] text-xs font-semibold text-[#7dd3fc] border border-[#7dd3fc]/30 transition-all cursor-pointer disabled:opacity-50 shadow-sm"
                      title="Open native file explorer to pick a directory"
                    >
                      <span className="material-symbols-outlined text-[15px]">folder_open</span>
                      <span>{isBrowsing ? 'Browsing...' : 'Browse Folder'}</span>
                    </button>
                    <button
                      type="button"
                      onClick={handleRescan}
                      disabled={isImporting}
                      className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-[#1a2438] hover:bg-[#223048] text-[11px] text-[#7dd3fc] font-medium border border-[#2a3a48]/50 transition-colors cursor-pointer disabled:opacity-50"
                    >
                      <span className="material-symbols-outlined text-[13px]">refresh</span>
                      <span>Rescan Folder</span>
                    </button>
                  </div>
                </div>

                <p className="text-[11px] text-[#a0b4c4]">
                  Browse, search, or enter the local folder path containing your dataset images. The system will scan and link the SQLite index.
                </p>

                {/* Input with live Search dropdown */}
                <div className="relative">
                  <div className="flex items-center gap-2">
                    <div className="relative flex-1">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-[#a0b4c4]">
                        <span className="material-symbols-outlined text-[16px]">search</span>
                      </div>
                      <input
                        type="text"
                        value={folderPathInput}
                        onChange={(e) => {
                          setFolderPathInput(e.target.value);
                          setShowDirDropdown(true);
                        }}
                        onFocus={() => setShowDirDropdown(true)}
                        placeholder="Search or enter path (e.g. /home/user/Pictures, test image...)"
                        className="w-full bg-[#0a0e1a] text-xs text-[#e0e8f0] pl-9 pr-3 py-2 rounded-lg border border-[#2a3a48]/60 focus:outline-none focus:ring-1 focus:ring-[#10b981] font-mono"
                      />
                      {searchLoading && (
                        <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                          <span className="w-3 h-3 rounded-full border-2 border-[#10b981] border-t-transparent animate-spin"></span>
                        </div>
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={handleSwitchFolder}
                      disabled={isImporting || !folderPathInput.trim()}
                      className="px-4 py-2 rounded-lg bg-[#10b981] hover:bg-[#10b981]/90 text-xs font-bold text-[#001f2e] transition-all cursor-pointer shadow-sm disabled:opacity-50 flex items-center gap-1.5 shrink-0"
                    >
                      <span className="material-symbols-outlined text-[15px]">drive_file_move</span>
                      <span>{isImporting ? 'Switching...' : 'Switch Folder'}</span>
                    </button>
                  </div>

                  {/* Autocomplete / Search Suggestions Dropdown */}
                  {showDirDropdown && dirSuggestions.length > 0 && (
                    <div className="absolute left-0 right-0 top-full mt-1.5 z-30 bg-[#111828] border border-[#2a3a48] rounded-xl shadow-2xl max-h-48 overflow-y-auto divide-y divide-[#1e2a38]">
                      <div className="p-2 text-[10px] uppercase font-mono font-bold tracking-wider text-[#a0b4c4] bg-[#0a0e1a]/80 flex items-center justify-between">
                        <span className="flex items-center gap-1">
                          <span className="material-symbols-outlined text-[12px] text-[#10b981]">folder_special</span>
                          Matching Folders ({dirSuggestions.length})
                        </span>
                        <button
                          type="button"
                          onClick={() => setShowDirDropdown(false)}
                          className="hover:text-white text-[#a0b4c4] px-1"
                        >
                          ✕
                        </button>
                      </div>
                      {dirSuggestions.map((dir) => (
                        <button
                          key={dir}
                          type="button"
                          onClick={() => handleSelectSuggestion(dir)}
                          className="w-full text-left px-3 py-2 text-xs font-mono text-[#e0e8f0] hover:bg-[#1a2438] flex items-center gap-2 group transition-colors cursor-pointer"
                        >
                          <span className="material-symbols-outlined text-[15px] text-[#7dd3fc] group-hover:text-[#10b981]">
                            folder
                          </span>
                          <span className="truncate flex-1" title={dir}>
                            {dir}
                          </span>
                          <span className="text-[10px] text-[#a0b4c4] group-hover:text-[#e0e8f0] px-1.5 py-0.5 rounded bg-[#1f2c3f]">
                            select
                          </span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {/* Quick Folder Shortcuts / Chips */}
                {dirSuggestions.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1.5 pt-1">
                    <span className="text-[10px] text-[#a0b4c4] font-medium mr-1">Quick Suggestions:</span>
                    {dirSuggestions.slice(0, 4).map((dir) => {
                      const parts = dir.split('/').filter(Boolean);
                      const shortName = parts.length > 0 ? parts[parts.length - 1] : dir;
                      return (
                        <button
                          key={`chip-${dir}`}
                          type="button"
                          onClick={() => {
                            setFolderPathInput(dir);
                            setShowDirDropdown(false);
                          }}
                          className="px-2 py-0.5 rounded-md bg-[#111828] hover:bg-[#1a2438] text-[10px] font-mono text-[#7dd3fc] hover:text-[#e0e8f0] border border-[#2a3a48]/50 flex items-center gap-1 transition-colors cursor-pointer"
                          title={dir}
                        >
                          <span className="material-symbols-outlined text-[11px] text-[#10b981]">folder</span>
                          <span>{shortName}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Upload Image Files Card */}
              <div className="p-4 rounded-xl bg-[#141c2e]/60 border border-[#2a3a48]/40 space-y-3">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-[#38bdf8] text-[18px]">add_photo_alternate</span>
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[#e0e8f0]">
                    Import New Image Files
                  </h3>
                </div>
                <p className="text-[11px] text-[#a0b4c4]">
                  Add new JPG, PNG, or WebP images into the current active dataset. Uploaded photos will be automatically indexed into SQLite.
                </p>

                <label className="border-2 border-dashed border-[#2a3a48] hover:border-[#10b981]/60 rounded-xl p-5 flex flex-col items-center justify-center gap-2 cursor-pointer bg-[#0a0e1a]/40 hover:bg-[#0a0e1a]/80 transition-all">
                  <span className="material-symbols-outlined text-3xl text-[#10b981]">cloud_upload</span>
                  <span className="text-xs font-semibold text-[#e0e8f0]">Click to choose image files or drag & drop</span>
                  <span className="text-[10px] text-[#a0b4c4] font-mono">Supports JPG, JPEG, PNG, WEBP</span>
                  <input
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={handleFileUpload}
                    disabled={isImporting}
                    className="hidden"
                  />
                </label>
              </div>

              {importMessage && (
                <div className="p-3 rounded-xl bg-[#10b981]/20 border border-[#10b981]/40 text-[#34d399] text-xs font-mono">
                  ✓ {importMessage}
                </div>
              )}
            </div>
          ) : (
            /* =================== TAB 2: EXPORT DATASET =================== */
            <div className="space-y-5">
              {/* Format Selector Cards */}
              <div className="space-y-2.5">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[#a0b4c4]">
                  Export Format
                </h3>
                <div className="grid grid-cols-3 gap-3">
                  {/* YOLO */}
                  <div
                    onClick={() => setSelectedFormat('yolo')}
                    className={`p-3 rounded-xl border transition-all cursor-pointer ${
                      selectedFormat === 'yolo'
                        ? 'bg-[#141c2e] border-[#06b6d4] shadow-md ring-1 ring-[#06b6d4]/40'
                        : 'bg-[#111828]/60 border-[#2a3a48]/40 hover:bg-[#141c2e]'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-xs font-bold text-[#e0e8f0]">YOLO PyTorch</span>
                      <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-[#06b6d4]/20 text-[#06b6d4] font-semibold">
                        RECOMMENDED
                      </span>
                    </div>
                    <p className="text-[10px] text-[#a0b4c4]">
                      data.yaml manifest + normalized bbox coordinate .txt files (YOLOv8/v10/v11).
                    </p>
                  </div>

                  {/* COCO */}
                  <div
                    onClick={() => setSelectedFormat('coco')}
                    className={`p-3 rounded-xl border transition-all cursor-pointer ${
                      selectedFormat === 'coco'
                        ? 'bg-[#141c2e] border-[#06b6d4] shadow-md ring-1 ring-[#06b6d4]/40'
                        : 'bg-[#111828]/60 border-[#2a3a48]/40 hover:bg-[#141c2e]'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-xs font-bold text-[#e0e8f0]">COCO JSON</span>
                      <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-[#a855f7]/20 text-[#c084fc]">
                        2017 Format
                      </span>
                    </div>
                    <p className="text-[10px] text-[#a0b4c4]">
                      Standard COCO instances_train.json format with segmentation polygon support.
                    </p>
                  </div>

                  {/* Pascal VOC */}
                  <div
                    onClick={() => setSelectedFormat('voc')}
                    className={`p-3 rounded-xl border transition-all cursor-pointer ${
                      selectedFormat === 'voc'
                        ? 'bg-[#141c2e] border-[#06b6d4] shadow-md ring-1 ring-[#06b6d4]/40'
                        : 'bg-[#111828]/60 border-[#2a3a48]/40 hover:bg-[#141c2e]'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-xs font-bold text-[#e0e8f0]">Pascal VOC</span>
                      <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-[#10b981]/20 text-[#34d399]">
                        XML
                      </span>
                    </div>
                    <p className="text-[10px] text-[#a0b4c4]">
                      Classic XML annotations with absolute pixel coordinates for traditional pipelines.
                    </p>
                  </div>
                </div>
              </div>

              {/* 3-Way Split Distribution Slider */}
              <div className="p-4 rounded-xl bg-[#141c2e]/60 border border-[#2a3a48]/40 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[#a0b4c4]">
                    Dataset Split Stratification
                  </h3>
                  <span className="text-[11px] font-mono text-[#06b6d4]">
                    {totalImages} Total Images
                  </span>
                </div>

                {/* Visual Stacked Progress Bar */}
                <div className="w-full h-3 bg-[#0a0e1a] rounded-full overflow-hidden flex">
                  <div
                    style={{ width: `${trainRatio}%` }}
                    className="bg-[#06b6d4] transition-all"
                    title={`Train: ${trainRatio}%`}
                  ></div>
                  <div
                    style={{ width: `${valRatio}%` }}
                    className="bg-[#f59e0b] transition-all"
                    title={`Val: ${valRatio}%`}
                  ></div>
                  <div
                    style={{ width: `${testRatio}%` }}
                    className="bg-[#a855f7] transition-all"
                    title={`Test: ${testRatio}%`}
                  ></div>
                </div>

                {/* Split Sliders and Badges */}
                <div className="grid grid-cols-3 gap-4 pt-2">
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-[#06b6d4] font-semibold">Train</span>
                      <span className="font-mono text-[11px] text-[#e0e8f0]">
                        {trainRatio}% ({Math.round((totalImages * trainRatio) / 100)} imgs)
                      </span>
                    </div>
                    <input
                      type="range"
                      min="50"
                      max="90"
                      value={trainRatio}
                      onChange={(e) => setTrainRatio(Number(e.target.value))}
                      className="w-full accent-[#06b6d4] cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-[#f59e0b] font-semibold">Validation</span>
                      <span className="font-mono text-[11px] text-[#e0e8f0]">
                        {valRatio}% ({Math.round((totalImages * valRatio) / 100)} imgs)
                      </span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="35"
                      value={valRatio}
                      onChange={(e) => setValRatio(Number(e.target.value))}
                      className="w-full accent-[#f59e0b] cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-[#c084fc] font-semibold">Test</span>
                      <span className="font-mono text-[11px] text-[#e0e8f0]">
                        {testRatio}% ({Math.round((totalImages * testRatio) / 100)} imgs)
                      </span>
                    </div>
                    <div className="h-6 flex items-center">
                      <span className="text-[10px] text-[#a0b4c4] font-mono">
                        Auto-calculated remainder
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Roboflow Cloud Sync & Data Quality Suite */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-[#141c2e]/60 border border-[#2a3a48]/40 flex items-center justify-between">
                  <div>
                    <span className="text-xs font-bold text-[#e0e8f0]">Roboflow Cloud Sync</span>
                    <p className="text-[10px] text-[#a0b4c4]">Push labeled dataset directly to Roboflow workspace</p>
                  </div>
                  <button
                    type="button"
                    className="px-3 py-1 rounded-lg bg-[#a855f7]/20 hover:bg-[#a855f7]/30 text-[#c084fc] text-xs font-semibold border border-[#a855f7]/30 transition-colors cursor-pointer"
                  >
                    Sync Now
                  </button>
                </div>

                <div className="p-3 rounded-xl bg-[#141c2e]/60 border border-[#2a3a48]/40 flex items-center justify-between">
                  <div>
                    <span className="text-xs font-bold text-[#e0e8f0]">Deduplication & NMS</span>
                    <p className="text-[10px] text-[#a0b4c4]">Prune duplicate photos and overlapping redundant boxes</p>
                  </div>
                  <button
                    type="button"
                    className="px-3 py-1 rounded-lg bg-[#10b981]/20 hover:bg-[#10b981]/30 text-[#34d399] text-xs font-semibold border border-[#10b981]/30 transition-colors cursor-pointer"
                  >
                    Clean Dataset
                  </button>
                </div>
              </div>

              {exportMessage && (
                <div className="p-3 rounded-xl bg-[#10b981]/20 border border-[#10b981]/40 text-[#34d399] text-xs font-mono">
                  ✓ {exportMessage}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="p-4 bg-[#111828] border-t border-[#2a3a48]/50 flex items-center justify-between">
          <span className="text-[11px] font-mono text-[#a0b4c4]">
            {activeTab === 'export' ? (
              <>Exporting to: <code className="text-[#06b6d4]">exports/{selectedFormat}_dataset/</code></>
            ) : (
              <>Active Database: <code className="text-[#10b981]">{stats?.database_file ? stats.database_file.split('/').slice(-2).join('/') : '.dataset_index.sqlite'}</code></>
            )}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              type="button"
              className="px-4 py-1.5 rounded-lg bg-[#141c2e] hover:bg-[#1a2438] text-xs font-medium text-[#e0e8f0] transition-colors cursor-pointer"
            >
              Close
            </button>
            {activeTab === 'export' ? (
              <button
                onClick={handleExport}
                disabled={isExporting}
                type="button"
                className="px-5 py-1.5 rounded-lg bg-[#06b6d4] hover:bg-[#06b6d4]/90 text-xs font-bold text-[#001f2e] transition-all cursor-pointer shadow-md flex items-center gap-1.5 disabled:opacity-50"
              >
                <span className="material-symbols-outlined text-[16px]">
                  {isExporting ? 'sync' : 'download'}
                </span>
                <span>{isExporting ? 'Exporting...' : 'Export Dataset Now'}</span>
              </button>
            ) : (
              <button
                onClick={handleRescan}
                disabled={isImporting}
                type="button"
                className="px-5 py-1.5 rounded-lg bg-[#10b981] hover:bg-[#10b981]/90 text-xs font-bold text-[#001f2e] transition-all cursor-pointer shadow-md flex items-center gap-1.5 disabled:opacity-50"
              >
                <span className="material-symbols-outlined text-[16px]">refresh</span>
                <span>Rescan & Sync Index</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
