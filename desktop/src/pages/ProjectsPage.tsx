import React, { useState } from 'react';
import type { ImageMeta, Page, SystemHealth } from '../types';
import { getImageUrl, imageIdentity, selectDatasetFolder } from '../services/api';
import { Badge, Button, Card, Icon, PageHeader, ProgressBar } from '../components/ui';
import { formatCount } from '../format';
import { datasetNameFrom, summarizeDataset } from './datasetSummary';
import { forgetDataset, recentDatasets } from './recentDatasets';

interface Props {
  images: ImageMeta[];
  health: SystemHealth | null;
  datasetName: string;
  onNavigate: (page: Page) => void;
  onDatasetChanged: () => void;
  onCloseProject: (folder: string) => Promise<void>;
}

export const ProjectsPage: React.FC<Props> = ({ images, health, datasetName, onNavigate, onDatasetChanged, onCloseProject }) => {
  const summary = summarizeDataset(images);
  const [query, setQuery] = useState('');
  const [recent, setRecent] = useState<string[]>(recentDatasets);
  const [opening, setOpening] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pendingRemoval, setPendingRemoval] = useState<string | null>(null);
  const [removing, setRemoving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const current = health?.dataset_dir || '';
  const matches = (folder: string) => !query || folder.toLowerCase().includes(query.toLowerCase());
  const others = recent.filter(folder => folder !== current);
  const shownOthers = others.filter(matches);
  const showCurrent = !!current && (!query || datasetName.toLowerCase().includes(query.toLowerCase()));

  const removeFolder = async (folder: string) => {
    setRemoving(true);
    setError(null);
    try {
      if (folder === current) await onCloseProject(folder);
      if (!forgetDataset(folder)) throw new Error('Daftar proyek tidak bisa disimpan. Coba hapus lagi.');
      setRecent(recentDatasets());
      setPendingRemoval(null);
      setMessage(`Proyek ${datasetNameFrom(folder)} dihapus dari daftar. File gambar dan anotasi tetap tersimpan.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRemoving(false);
    }
  };

  const removalAction = (folder: string) => pendingRemoval === folder ? (
    <div className="mt-3 w-full rounded-lg border border-line-strong bg-brand-50 p-3"
      role="group" aria-label={`Konfirmasi hapus proyek ${datasetNameFrom(folder)}`}>
      <p className="text-sm font-semibold text-ink-900">Hapus {datasetNameFrom(folder)} dari daftar proyek?</p>
      <p className="mt-1 text-xs leading-relaxed text-ink-500">
        {folder === current && 'Proyek aktif akan ditutup. '}File gambar dan anotasi tetap tersimpan di folder.
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        <Button size="sm" variant="secondary" disabled={removing} onClick={() => setPendingRemoval(null)}>Batal</Button>
        <Button size="sm" variant="danger" disabled={removing} onClick={() => void removeFolder(folder)}>
          {removing ? 'Menghapus…' : 'Hapus proyek'}
        </Button>
      </div>
    </div>
  ) : (
    <Button size="sm" variant="danger" icon="delete" disabled={removing || opening !== null}
      aria-label={`Hapus proyek ${datasetNameFrom(folder)}`}
      onClick={() => { setPendingRemoval(folder); setMessage(null); setError(null); }}>Hapus proyek</Button>
  );

  const openFolder = async (folder: string) => {
    setOpening(folder);
    setError(null);
    try {
      await selectDatasetFolder(folder);
      onDatasetChanged();
      onNavigate('overview');
    } catch (err) {
      setError(`Folder tidak bisa dibuka: ${err instanceof Error ? err.message : String(err)}`);
      forgetDataset(folder);
      setRecent(recentDatasets());
    } finally {
      setOpening(null);
    }
  };

  return (
    <div className="mx-auto max-w-[1040px] px-8 py-10">
      <PageHeader
        title="Proyek kamu"
        subtitle="Dataset, anotasi, dan progres review dalam satu tempat."
        actions={<Button icon="add" onClick={() => onNavigate('import')}>Impor dataset</Button>}
      />

      <div className="mb-6 flex items-center gap-4">
        <label className="flex h-10 w-[320px] items-center gap-2 rounded-lg border border-line-strong bg-white px-3 focus-within:border-brand-500">
          <Icon name="search" className="text-[18px] text-ink-400" />
          <input
            value={query}
            onChange={event => setQuery(event.target.value)}
            placeholder="Cari proyek…"
            aria-label="Cari proyek"
            className="w-full bg-transparent text-sm text-ink-900 placeholder:text-ink-400 focus:outline-none"
          />
        </label>
        <span className="text-xs text-ink-500">{Number(!!current) + others.length} proyek lokal</span>
      </div>

      {error && <p role="alert" className="mb-4 text-sm text-danger-600">{error}</p>}
      {message && <p role="status" className="mb-4 text-sm text-ok-600">{message}</p>}
      {!current && others.length === 0 && (
        <p className="mb-5 text-sm text-ink-500">Belum ada proyek. Impor folder untuk mulai membuat anotasi.</p>
      )}

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        {showCurrent && (
          <Card className="overflow-hidden">
            <div className="relative h-[178px] bg-canvas">
              {images[0] && (
                <img src={getImageUrl(imageIdentity(images[0]))} alt="" className="h-full w-full object-cover" />
              )}
              <span className="absolute right-3 top-3 rounded-md bg-white/95 px-2 py-1 text-[11px] font-medium text-ink-700 shadow-sm">
                Aktif
              </span>
            </div>
            <div className="p-5">
              <h2 className="truncate text-xl font-bold text-ink-900" title={current}>{datasetName || 'Dataset'}</h2>
              <div className="mt-1 text-xs text-ink-500">
                Deteksi objek <span className="mx-2 text-ink-300">/</span> {health?.classes?.length || 0} kelas
              </div>
              <div className="mt-5 flex gap-6 text-sm">
                <span><b className="text-ink-900">{formatCount(summary.total)}</b> <span className="text-xs text-ink-500">gambar</span></span>
                <span><b className="text-ink-900">{formatCount(summary.annotations)}</b> <span className="text-xs text-ink-500">anotasi</span></span>
              </div>
              <div className="mt-5 flex items-center justify-between text-xs text-ink-500">
                <span>Review</span>
                <span className="tabular-nums">{formatCount(summary.approved)}/{formatCount(summary.total)} gambar</span>
              </div>
              <ProgressBar value={summary.approvedPercent} className="mt-2" />
              <div className="mt-5 flex flex-wrap items-center gap-2">
                <Button variant="secondary" disabled={removing} onClick={() => onNavigate('review')}>Buka proyek</Button>
                {removalAction(current)}
              </div>
            </div>
          </Card>
        )}

        <Card tone="soft" className="flex flex-col justify-center border-dashed p-8">
          <span className="flex h-11 w-11 items-center justify-center rounded-lg bg-brand-100 text-brand-600">
            <Icon name="add_photo_alternate" className="text-[22px]" />
          </span>
          <h2 className="mt-5 text-xl font-bold leading-snug text-ink-900">Dataset berikutnya<br />dimulai dari gambarmu.</h2>
          <p className="mt-3 text-xs leading-relaxed text-ink-500">
            Pilih folder berisi JPG, PNG, atau WEBP. Label YOLO (.txt) yang sudah ada ikut terbaca.
          </p>
          <div className="mt-5">
            <Button variant="secondary" onClick={() => onNavigate('import')}>Impor dataset</Button>
          </div>
        </Card>
      </div>

      {shownOthers.length > 0 && (
        <section className="mt-10">
          <h2 className="mb-3 text-[15px] font-semibold text-ink-900">Terakhir dibuka</h2>
          <div className="divide-y divide-line overflow-hidden rounded-xl border border-line">
            {shownOthers.map(folder => (
              <div key={folder} className="flex flex-wrap items-center gap-3 px-4 py-3">
                <Icon name="folder" className="text-[20px] text-ink-400" />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-semibold text-ink-900">{datasetNameFrom(folder)}</div>
                  <div className="truncate text-xs text-ink-500" title={folder}>{folder}</div>
                </div>
                <Badge>Lokal</Badge>
                <Button size="sm" variant="secondary" disabled={opening !== null || removing} onClick={() => void openFolder(folder)}>
                  {opening === folder ? 'Membuka…' : 'Buka'}
                </Button>
                {removalAction(folder)}
              </div>
            ))}
          </div>
        </section>
      )}

      <p className="mt-6 text-[11px] text-ink-500">
        Anotasi disimpan di folder dataset itu sendiri, jadi proyek tetap ada setelah aplikasi ditutup.
      </p>
    </div>
  );
};
