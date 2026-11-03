import React, { useEffect, useState } from 'react';
import type { ImageMeta, Page, SystemHealth } from '../types';
import {
  browseDatasetFolder,
  getImageUrl,
  imageIdentity,
  registerDatasetClass,
  rescanDataset,
  searchDatasetDirectories,
  selectDatasetFolder,
  uploadDatasetImages,
} from '../services/api';
import { Badge, Button, Card, Icon, Note, PageHeader } from '../components/ui';
import { formatCount } from '../format';
import { WorkflowSteps } from './WorkflowSteps';

interface Props {
  images: ImageMeta[];
  health: SystemHealth | null;
  datasetName: string;
  onNavigate: (page: Page) => void;
  onDatasetChanged: () => void;
  onClassesChanged: (classes: string[]) => void;
}

const IMAGE_TYPES = /\.(jpe?g|png|webp)$/i;

export const ImportPage: React.FC<Props> = ({
  images, health, datasetName, onNavigate, onDatasetChanged, onClassesChanged,
}) => {
  const [folder, setFolder] = useState(health?.dataset_dir || '');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [message, setMessage] = useState<{ tone: 'ok' | 'error'; text: string } | null>(null);
  const [newClass, setNewClass] = useState('');

  useEffect(() => { if (health?.dataset_dir) setFolder(health.dataset_dir); }, [health?.dataset_dir]);

  useEffect(() => {
    if (!showSuggestions) return;
    let active = true;
    const timer = setTimeout(() => {
      searchDatasetDirectories(folder)
        .then(result => { if (active) setSuggestions(result.directories || []); })
        .catch(() => { if (active) setSuggestions([]); });
    }, 200);
    return () => { active = false; clearTimeout(timer); };
  }, [folder, showSuggestions]);

  const run = async (work: () => Promise<string>) => {
    setBusy(true);
    setMessage(null);
    try {
      setMessage({ tone: 'ok', text: await work() });
      onDatasetChanged();
    } catch (err) {
      setMessage({ tone: 'error', text: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusy(false);
    }
  };

  const openFolder = (path: string) => run(async () => {
    const result = await selectDatasetFolder(path.trim());
    setShowSuggestions(false);
    return `Folder aktif: ${result.active_directory} (${formatCount(result.indexed_count)} gambar)`;
  });

  const browse = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const result = await browseDatasetFolder();
      if (result.status === 'ok' && result.active_directory) {
        setFolder(result.active_directory);
        setMessage({ tone: 'ok', text: `Folder aktif: ${result.active_directory} (${formatCount(result.indexed_count || 0)} gambar)` });
        onDatasetChanged();
      } else if (result.status === 'unsupported') {
        setMessage({ tone: 'error', text: 'Dialog folder tidak tersedia di sistem ini. Ketik path folder lalu pilih Buka folder.' });
      }
    } catch (err) {
      setMessage({ tone: 'error', text: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusy(false);
    }
  };

  const upload = (files: FileList | File[]) => {
    const accepted = Array.from(files).filter(file => IMAGE_TYPES.test(file.name));
    if (!accepted.length) {
      setMessage({ tone: 'error', text: 'Pilih file JPG, PNG, atau WEBP.' });
      return;
    }
    void run(async () => {
      const result = await uploadDatasetImages(accepted);
      return `${formatCount(result.count)} gambar ditambahkan. Total sekarang ${formatCount(result.total_indexed)} gambar.`;
    });
  };

  const addClass = async () => {
    const name = newClass.trim();
    if (!name) return;
    try {
      onClassesChanged(await registerDatasetClass(name, health?.dataset_dir));
      setNewClass('');
    } catch (err) {
      setMessage({ tone: 'error', text: `Kelas tidak bisa ditambahkan: ${err instanceof Error ? err.message : String(err)}` });
    }
  };

  return (
    <div className="mx-auto max-w-[1040px] px-8 py-10">
      <PageHeader title="Mulai dari dataset kamu" subtitle="Impor gambar lokal, lalu tentukan label yang ingin dideteksi." />
      <WorkflowSteps current="import" onNavigate={onNavigate} />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_352px]">
        <Card className="p-6">
          <label htmlFor="dataset-folder" className="text-[13px] font-semibold text-ink-900">Folder dataset</label>
          <div className="relative mt-2 flex gap-2">
            <input
              id="dataset-folder"
              value={folder}
              onChange={event => { setFolder(event.target.value); setShowSuggestions(true); }}
              onFocus={() => setShowSuggestions(true)}
              onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
              onKeyDown={event => { if (event.key === 'Enter' && folder.trim()) void openFolder(folder); }}
              placeholder="/home/kamu/Dataset/traffic"
              className="h-11 min-w-0 flex-1 rounded-lg border border-line-strong px-3 text-sm text-ink-900 placeholder:text-ink-400 focus:border-brand-500 focus:outline-none"
            />
            <Button variant="secondary" disabled={busy || !folder.trim()} onClick={() => void openFolder(folder)}>Buka folder</Button>
            {showSuggestions && suggestions.length > 0 && (
              <ul className="absolute left-0 right-0 top-12 z-20 max-h-56 overflow-y-auto rounded-lg border border-line bg-white py-1 shadow-lg">
                {suggestions.map(path => (
                  <li key={path}>
                    <button
                      type="button"
                      onMouseDown={event => { event.preventDefault(); setFolder(path); void openFolder(path); }}
                      className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-ink-700 hover:bg-brand-50 cursor-pointer"
                    >
                      <Icon name="folder" className="text-[18px] text-ink-400" />
                      <span className="truncate">{path}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="mt-3 flex gap-2">
            <Button size="sm" variant="ghost" icon="folder_open" disabled={busy} onClick={() => void browse()}>Pilih folder…</Button>
            <Button size="sm" variant="ghost" icon="refresh" disabled={busy} onClick={() => void run(async () => {
              const result = await rescanDataset();
              return `Pindai ulang selesai: ${formatCount(result.indexed_count)} gambar.`;
            })}>Pindai ulang</Button>
          </div>

          <label
            onDragOver={event => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={event => { event.preventDefault(); setDragging(false); upload(event.dataTransfer.files); }}
            className={`mt-5 flex flex-col items-center justify-center rounded-xl border border-dashed px-6 py-10 text-center transition-colors cursor-pointer ${
              dragging ? 'border-brand-600 bg-brand-100' : 'border-brand-300 bg-brand-50 hover:bg-brand-100'
            }`}
          >
            <Icon name="image" className="text-[30px] text-brand-500" />
            <span className="mt-3 text-base font-semibold text-ink-900">Tarik gambar ke sini</span>
            <span className="mt-1.5 text-[11px] text-ink-500">JPG, PNG, atau WEBP · ditambahkan ke folder dataset aktif</span>
            <span className="mt-4 inline-flex h-10 items-center rounded-lg border border-line-strong bg-white px-4 text-sm font-semibold text-ink-900">
              Pilih gambar
            </span>
            <input
              type="file"
              multiple
              accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
              disabled={busy}
              onChange={event => { if (event.target.files) upload(event.target.files); event.target.value = ''; }}
              className="hidden"
            />
          </label>

          <div className="mt-5 flex items-center gap-3">
            {images[0] ? (
              <img src={getImageUrl(imageIdentity(images[0]))} alt="" className="h-10 w-14 rounded object-cover" />
            ) : <span className="h-10 w-14 rounded bg-brand-100" />}
            <div className="min-w-0 flex-1">
              <div className="text-[13px] font-semibold text-ink-900">{formatCount(images.length)} gambar tersedia</div>
              <div className="truncate text-xs text-ink-500" title={health?.dataset_dir}>{datasetName || 'Belum ada folder aktif'}</div>
            </div>
            <Badge tone={images.length ? 'brand' : 'neutral'}>{busy ? 'Memproses…' : images.length ? 'Siap' : 'Kosong'}</Badge>
          </div>

          {message && (
            <p role={message.tone === 'error' ? 'alert' : 'status'}
              className={`mt-4 text-xs ${message.tone === 'error' ? 'text-danger-600' : 'text-ok-600'}`}>
              {message.text}
            </p>
          )}
        </Card>

        <Card className="h-fit p-6">
          <h2 className="text-lg font-bold text-ink-900">Kelas objek</h2>
          <div className="mt-4 rounded-md bg-brand-100 px-2 py-1.5 text-xs font-medium text-brand-700">
            {datasetName || 'Dataset aktif'}
          </div>
          <div className="mt-5 text-xs text-ink-500">Kelas terdaftar</div>
          <div className="mt-2 flex flex-wrap gap-2">
            {(health?.classes || []).map(name => <Badge key={name}>{name}</Badge>)}
            {!health?.classes?.length && <span className="text-xs text-ink-400">Belum ada kelas.</span>}
          </div>
          <div className="mt-3 flex gap-2">
            <input
              value={newClass}
              onChange={event => setNewClass(event.target.value)}
              onKeyDown={event => { if (event.key === 'Enter') void addClass(); }}
              placeholder="Kelas baru, mis. helmet"
              aria-label="Nama kelas baru"
              className="h-9 min-w-0 flex-1 rounded-lg border border-line-strong px-3 text-sm placeholder:text-ink-400 focus:border-brand-500 focus:outline-none"
            />
            <Button size="sm" variant="secondary" disabled={!newClass.trim()} onClick={() => void addClass()}>Tambah</Button>
          </div>
          <div className="mt-6 text-xs text-ink-500">Jenis anotasi</div>
          <div className="mt-1.5 text-[15px] font-semibold text-ink-900">Bounding box &amp; poligon</div>
          <Note className="mt-6" title="Privat sejak awal">
            Gambar diproses di perangkatmu. Tidak ada upload ke cloud kecuali kamu memilih inference cloud atau Colab.
          </Note>
        </Card>
      </div>

      <div className="mt-6 flex items-center justify-between">
        <span className="text-xs text-ink-500">Impor tidak menjalankan model AI.</span>
        <Button trailingIcon="arrow_forward" disabled={!images.length} onClick={() => onNavigate('autolabel')}>Lanjut ke auto-label</Button>
      </div>
    </div>
  );
};
