import React, { useEffect, useRef, useState } from 'react';
import {
  fetchAnnotationImportStatus,
  startAnnotationImport,
  type AnnotationImportStatus,
} from '../services/api';
import { Badge, Button, Card, Icon, Note, ProgressBar } from '../components/ui';
import { formatCount } from '../format';
import { useAnnotationSources } from './useAnnotationSources';

interface Props {
  datasetKey: string | undefined;
  /** Called once when an import finishes, so image lists and counts can be refreshed. */
  onImported: () => void;
  onOpenReview: () => void;
}

// Survives the card being unmounted, so finishing an import while on another page still refreshes once.
let refreshedJobId = '';

/** Load a dataset's existing COCO annotation files into Anotara so they can be reviewed and audited. */
export const AnnotationImportCard: React.FC<Props> = ({ datasetKey, onImported, onOpenReview }) => {
  const [reload, setReload] = useState(0);
  const { sources, error } = useAnnotationSources(datasetKey, reload);
  const [job, setJob] = useState<AnnotationImportStatus | null>(null);
  const [overwrite, setOverwrite] = useState(false);
  const [customPath, setCustomPath] = useState('');
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const callback = useRef(onImported);
  callback.current = onImported;

  // Pick up a job that is already running, or just finished, when the page opens.
  useEffect(() => {
    let active = true;
    fetchAnnotationImportStatus().then(status => { if (active) setJob(status); }).catch(() => { /* none yet */ });
    return () => { active = false; };
  }, [datasetKey]);

  useEffect(() => {
    if (!job?.running) return;
    const timer = setInterval(() => {
      fetchAnnotationImportStatus().then(setJob).catch(() => { /* keep the last status */ });
    }, 700);
    return () => clearInterval(timer);
  }, [job?.running]);

  useEffect(() => {
    if (job?.completed && job.job_id && refreshedJobId !== job.job_id) {
      refreshedJobId = job.job_id;
      callback.current();
      setReload(value => value + 1);
    }
  }, [job]);

  const begin = async (paths: string[]) => {
    setStarting(true);
    setStartError(null);
    try {
      await startAnnotationImport({ sources: paths, overwrite, dataset_id: datasetKey });
      setJob(await fetchAnnotationImportStatus());
    } catch (err) {
      setStartError(err instanceof Error ? err.message : String(err));
    } finally {
      setStarting(false);
    }
  };

  const mine = !!job && (!job.dataset_id || job.dataset_id === datasetKey);
  const running = mine && !!job?.running;
  const finished = mine && !!job?.completed && !!job.job_id;
  const percent = job?.total ? Math.round(((job.processed || 0) / job.total) * 100) : 0;
  const totalBoxes = (sources || []).reduce((sum, source) => sum + source.annotations, 0);
  const totalImages = (sources || []).reduce((sum, source) => sum + source.images, 0);

  return (
    <Card className="mt-6 p-6">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-lg font-bold text-ink-900">Anotasi yang sudah ada</h2>
        <Badge tone="brand">COCO</Badge>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-ink-500">
        Gambar dan label biasanya datang terpisah. Impor file anotasi COCO supaya kotaknya muncul di review, audit, dan ekspor.
        Gambar tidak disalin; kotak yang bertumpuk tidak dibuang karena bisa jadi objek yang tertutup.
      </p>

      {sources === null && <p className="mt-4 text-xs text-ink-500">Mencari file anotasi di folder ini…</p>}
      {error && <p role="alert" className="mt-4 text-xs text-danger-600">Pencarian gagal: {error}</p>}

      {sources && sources.length > 0 && (
        <ul className="mt-4 divide-y divide-line rounded-lg border border-line">
          {sources.map(source => (
            <li key={source.path} className="flex items-start gap-3 px-4 py-3">
              <Icon name="description" className="mt-0.5 text-[20px] text-ink-400" />
              <div className="min-w-0 flex-1">
                <div className="truncate text-[13px] font-semibold text-ink-900" title={source.path}>{source.relative}</div>
                <div className="mt-0.5 text-xs text-ink-500">
                  {formatCount(source.images)} gambar · {formatCount(source.annotations)} kotak
                  {source.categories.length > 0 && <> · {source.categories.join(', ')}</>}
                </div>
                {source.found_images < source.images && (
                  <div className="mt-0.5 text-xs text-warn-700">
                    {formatCount(source.images - source.found_images)} gambar tidak ditemukan di {source.image_dir}/
                  </div>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
      {sources && sources.length === 0 && !error && (
        <p className="mt-4 text-xs text-ink-500">Tidak ada file anotasi COCO di folder ini. Kamu bisa menunjuk file lain di bawah.</p>
      )}

      {sources && sources.length > 0 && (
        <div className="mt-4 space-y-3">
          <label className="flex items-start gap-3 rounded-lg bg-brand-50 px-4 py-3 text-xs text-ink-700 cursor-pointer">
            <input type="checkbox" checked={overwrite} onChange={event => setOverwrite(event.target.checked)}
              className="mt-0.5 h-4 w-4 accent-brand-600" />
            <span>
              <span className="block text-[13px] font-semibold text-ink-900">Timpa anotasi yang sudah ada</span>
              Bawaan: gambar yang sudah punya label dilewati agar koreksimu aman.
            </span>
          </label>
          <Button block icon="download" disabled={running || starting}
            onClick={() => void begin(sources.map(source => source.path))}>
            {running ? `Mengimpor… ${percent}%` : `Impor ${formatCount(totalBoxes)} kotak dari ${formatCount(totalImages)} gambar`}
          </Button>
        </div>
      )}

      {running && job && (
        <div className="mt-4">
          <ProgressBar value={percent} />
          <div className="mt-2 truncate text-[11px] text-ink-500" title={job.current_file}>
            {formatCount(job.processed || 0)}/{formatCount(job.total || 0)} gambar · {formatCount(job.boxes || 0)} kotak
            {job.current_file ? ` · ${job.current_file}` : ''}
          </div>
        </div>
      )}

      {finished && job && !running && (
        <div className="mt-4 rounded-xl border border-brand-200 bg-brand-50 p-4" role="status">
          <div className="flex items-center gap-2 text-sm font-semibold text-ink-900">
            <Icon name={job.error ? 'error' : 'check_circle'} className={`text-[20px] ${job.error ? 'text-danger-600' : 'text-ok-600'}`} />
            {job.error ? 'Impor berhenti' : 'Impor selesai'}
          </div>
          <div className="mt-2 text-xs text-ink-700">
            <b>{formatCount(job.imported || 0)}</b> gambar diimpor · <b>{formatCount(job.boxes || 0)}</b> kotak
          </div>
          <ul className="mt-2 space-y-0.5 text-xs text-ink-500">
            {!!job.skipped_existing && <li>{formatCount(job.skipped_existing)} gambar dilewati karena sudah punya label</li>}
            {!!job.skipped_conflict && <li>{formatCount(job.skipped_conflict)} gambar dilewati karena sedang kamu edit</li>}
            {!!job.missing_images && <li>{formatCount(job.missing_images)} gambar tidak ditemukan di disk</li>}
            {!!job.invalid_boxes && <li>{formatCount(job.invalid_boxes)} kotak tidak valid dibuang</li>}
            {!!job.orphan_boxes && <li>{formatCount(job.orphan_boxes)} kotak merujuk gambar yang tidak ada</li>}
            {!!job.classes_added?.length && <li>Kelas baru ditambahkan: {job.classes_added.join(', ')}</li>}
            {!!job.failed_count && <li className="text-danger-600">{formatCount(job.failed_count)} gambar gagal diimpor</li>}
          </ul>
          {job.error && <p role="alert" className="mt-2 text-xs text-danger-600">{job.error}</p>}
          {!!job.imported && (
            <>
              <p className="mt-3 text-xs text-ink-500">Gambar yang diimpor berstatus “perlu review” sampai kamu menyetujuinya.</p>
              <Button size="sm" variant="secondary" className="mt-3" onClick={onOpenReview}>Buka antrean review</Button>
            </>
          )}
        </div>
      )}

      <div className="mt-5 border-t border-line pt-4">
        <label htmlFor="coco-path" className="text-[13px] font-semibold text-ink-900">Atau pilih file COCO lain</label>
        <div className="mt-2 flex gap-2">
          <input
            id="coco-path"
            value={customPath}
            onChange={event => setCustomPath(event.target.value)}
            onKeyDown={event => { if (event.key === 'Enter' && customPath.trim()) void begin([customPath.trim()]); }}
            placeholder="/path/ke/annotations.json"
            className="h-10 min-w-0 flex-1 rounded-lg border border-line-strong px-3 text-sm placeholder:text-ink-400 focus:border-brand-500 focus:outline-none"
          />
          <Button variant="secondary" disabled={!customPath.trim() || running || starting}
            onClick={() => void begin([customPath.trim()])}>Impor file ini</Button>
        </div>
      </div>
      {startError && <p role="alert" className="mt-3 text-xs text-danger-600">{startError}</p>}
      <Note className="mt-4" title="Disimpan di folder dataset">
        Anotasi ditulis ke <code className="font-mono">.visionlab/</code> dan label YOLO di folder <code className="font-mono">labels/</code>
        diperbarui. File COCO aslimu tidak diubah.
      </Note>
    </Card>
  );
};
