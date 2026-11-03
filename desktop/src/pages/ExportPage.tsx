import React, { useState } from 'react';
import type { ExportResult, ImageMeta, Page, SystemHealth } from '../types';
import { exportDataset, type ExportFormat } from '../services/api';
import { Badge, Button, Card, Icon, Note, PageHeader } from '../components/ui';
import { formatCount } from '../format';
import { summarizeDataset } from './datasetSummary';

interface Props {
  images: ImageMeta[];
  health: SystemHealth | null;
  datasetName: string;
  onNavigate: (page: Page) => void;
}

const FORMATS: { id: ExportFormat; label: string; detail: string; tree: string[]; splits: boolean }[] = [
  { id: 'yolo', label: 'YOLO', detail: 'images/ + labels/ + data.yaml', splits: true,
    tree: ['images/train · val · test', 'labels/train · val · test', 'data.yaml', 'dataset.yaml'] },
  { id: 'coco', label: 'COCO', detail: 'annotations.json', splits: true,
    tree: ['images/', 'annotations.json'] },
  { id: 'voc', label: 'Pascal VOC', detail: 'Satu XML per gambar', splits: false,
    tree: ['<nama-gambar>.xml', '…'] },
  { id: 'cvat', label: 'CVAT', detail: 'annotations.xml', splits: false,
    tree: ['annotations.xml'] },
];

export const ExportPage: React.FC<Props> = ({ images, health, datasetName, onNavigate }) => {
  const [format, setFormat] = useState<ExportFormat>('yolo');
  const [onlyApproved, setOnlyApproved] = useState(true);
  const [train, setTrain] = useState(80);
  const [val, setVal] = useState(10);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ExportResult | null>(null);

  const summary = summarizeDataset(images);
  const exportable = onlyApproved ? images.filter(image => image.status === 'reviewed') : images;
  const annotations = exportable.reduce((sum, image) => sum + (image.annotation_count || 0), 0);
  const test = Math.max(0, 100 - train - val);
  const active = FORMATS.find(item => item.id === format)!;
  const trainCount = Math.round((exportable.length * train) / 100);
  const valCount = Math.round((exportable.length * val) / 100);
  const splitCounts = [trainCount, valCount, Math.max(0, exportable.length - trainCount - valCount)];

  const runExport = async () => {
    setBusy(true);
    setError(null);
    try {
      setResult(await exportDataset({
        format, only_reviewed: onlyApproved,
        train_ratio: train / 100, val_ratio: val / 100, test_ratio: test / 100,
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  if (result) {
    const label = FORMATS.find(item => item.id === result.format)?.label || result.format.toUpperCase();
    return (
      <div className="mx-auto max-w-[1040px] px-8 py-10">
        <Card className="mx-auto max-w-[700px] px-10 py-12 text-center">
          <span className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-brand-100 text-brand-600">
            <Icon name="check_circle" className="text-[30px]" />
          </span>
          <Badge tone="brand" className="mt-5">Ekspor lokal selesai</Badge>
          <h1 className="mt-5 text-[28px] font-bold tracking-tight text-ink-900">Dataset kamu siap dilatih.</h1>
          <p className="mt-4 text-sm leading-relaxed text-ink-500">
            {formatCount(result.total_documents)} gambar telah dikemas dalam format {label}.
            {result.splits && FORMATS.find(item => item.id === result.format)?.splits && (
              <> Train {formatCount(result.splits.train)} · Val {formatCount(result.splits.val)} · Test {formatCount(result.splits.test)}.</>
            )}
          </p>
          <div className="mt-7 flex items-center gap-4 rounded-xl border border-line bg-brand-50 px-5 py-4 text-left">
            <Icon name="folder" className="text-[26px] text-ink-500" />
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-semibold text-ink-900" title={result.destination}>{result.destination}</div>
              <div className="mt-0.5 text-xs text-ink-500">{label} · {health?.classes?.length || 0} kelas · {['yolo', 'coco'].includes(result.format) ? 'gambar dan anotasi' : 'anotasi saja (gambar tidak disalin)'}</div>
            </div>
            <Button size="sm" variant="secondary" icon="content_copy"
              onClick={() => void navigator.clipboard?.writeText(result.destination)}>Salin path</Button>
          </div>
          <div className="mt-7 flex justify-center gap-3">
            <Button onClick={() => setResult(null)}>Ekspor lagi</Button>
            <Button variant="secondary" onClick={() => onNavigate('overview')}>Kembali ke proyek</Button>
          </div>
          <p className="mt-8 text-xs text-ink-500">File dibuat di perangkatmu. Periksa label hasil auto-label sebelum dipakai untuk training produksi.</p>
        </Card>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1040px] px-8 py-10">
      <PageHeader title="Ekspor dataset" subtitle="Folder ekspor berisi anotasi sesuai format pilihanmu. YOLO dan COCO juga menyalin gambarnya." />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_352px]">
        <Card className="p-6">
          <h2 className="text-lg font-bold text-ink-900">Format dataset</h2>
          <div className="mt-5 grid grid-cols-2 gap-3" role="radiogroup" aria-label="Format dataset">
            {FORMATS.map(item => {
              const selected = item.id === format;
              return (
                <button
                  key={item.id}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  onClick={() => setFormat(item.id)}
                  className={`flex items-start gap-3 rounded-xl border p-4 text-left transition-colors cursor-pointer ${
                    selected ? 'border-brand-600 bg-brand-50 ring-1 ring-brand-600' : 'border-line hover:border-brand-300'
                  }`}
                >
                  <Icon name={selected ? 'check_circle' : 'folder'} className={`text-[22px] ${selected ? 'text-brand-600' : 'text-ink-400'}`} />
                  <span>
                    <span className="block text-sm font-semibold text-ink-900">{item.label}</span>
                    <span className="mt-1 block text-xs text-ink-500">{item.detail}</span>
                  </span>
                </button>
              );
            })}
          </div>

          <label className="mt-5 flex items-center gap-3 rounded-lg bg-brand-50 px-4 py-3.5 text-sm text-ink-700 cursor-pointer">
            <input type="checkbox" checked={onlyApproved} onChange={event => setOnlyApproved(event.target.checked)}
              className="h-4 w-4 accent-brand-600" />
            Hanya ekspor gambar yang sudah disetujui
            <span className="ml-auto text-xs text-ink-500">{formatCount(summary.approved)} dari {formatCount(summary.total)}</span>
          </label>

          <h3 className="mt-7 text-[15px] font-semibold text-ink-900">Pembagian dataset</h3>
          {active.splits ? (
            <>
              <div className="mt-4 grid grid-cols-3 gap-3">
                {[
                  { label: 'Train', value: train, set: setTrain, min: 50, max: 95 },
                  { label: 'Validation', value: val, set: setVal, min: 0, max: Math.min(40, 100 - train) },
                  { label: 'Test', value: test, set: null, min: 0, max: 0 },
                ].map((split, index) => (
                  <div key={split.label} className="rounded-lg bg-brand-50 px-4 py-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-ink-700">{split.label}</span>
                      <span className="font-semibold tabular-nums text-ink-900">{split.set ? `${split.value}%` : 'sisa'}</span>
                    </div>
                    {split.set ? (
                      <input type="range" min={split.min} max={split.max} value={split.value}
                        aria-label={`Persentase ${split.label}`}
                        onChange={event => split.set!(Number(event.target.value))}
                        className="mt-2 w-full accent-brand-600 cursor-pointer" />
                    ) : <div className="mt-2 h-[22px] text-xs text-ink-500 leading-[22px]">{test}%</div>}
                    <div className="text-xs text-ink-500">{formatCount(splitCounts[index])} gambar</div>
                  </div>
                ))}
              </div>
              <p className="mt-4 text-xs text-ink-500">
                Pembagian acak tapi tetap sama di setiap ekspor, dan tidak menggandakan gambar. Pada dataset sangat kecil sebagian split bisa kosong.
              </p>
            </>
          ) : (
            <p className="mt-3 text-xs text-ink-500">{active.label} menulis satu set anotasi tanpa pembagian train/val/test.</p>
          )}

          <Note className="mt-6" title="Disimpan langsung di perangkatmu">
            Folder ekspor ditulis ke <code className="font-mono">exports/</code> tanpa mengirim gambar ke server mana pun.
          </Note>
        </Card>

        <Card className="flex h-fit flex-col p-6">
          <div className="rounded-md bg-[#eef1f6] px-3 py-1.5 text-xs text-ink-700">Paket ekspor</div>
          <h2 className="mt-5 truncate text-lg font-bold text-ink-900">{datasetName || 'Dataset'}</h2>
          <div className="mt-4 divide-y divide-transparent">
            {[
              [onlyApproved ? 'Gambar disetujui' : 'Gambar', formatCount(exportable.length)],
              ['Total anotasi', formatCount(annotations)],
              ['Kelas objek', `${health?.classes?.length || 0} kelas`],
              ['Format', active.label],
            ].map(([label, value]) => (
              <div key={label} className="flex items-center justify-between py-2.5 text-sm">
                <span className="text-ink-500">{label}</span>
                <span className="font-semibold text-ink-900">{value}</span>
              </div>
            ))}
          </div>
          <div className="mt-4 rounded-lg border border-line bg-brand-50 px-4 py-3 font-mono text-xs leading-6 text-ink-500">
            <div>exports/{format}_&lt;waktu&gt;/</div>
            {active.tree.map((entry, index) => (
              <div key={entry} className={`pl-3 ${index === 0 ? 'font-semibold text-brand-600' : ''}`}>{entry}</div>
            ))}
          </div>
          {error && <p role="alert" className="mt-4 text-xs text-danger-600">{error}</p>}
          <Button block className="mt-5" disabled={busy || exportable.length === 0} onClick={() => void runExport()}>
            {busy ? 'Membuat paket…' : 'Buat paket ekspor'}
          </Button>
          {exportable.length === 0 && (
            <p className="mt-3 text-xs text-ink-500">
              {onlyApproved ? 'Belum ada gambar yang disetujui. Setujui gambar di halaman review dulu.' : 'Dataset masih kosong.'}
            </p>
          )}
          <p className="mt-5 text-xs leading-relaxed text-ink-500">Ekspor tidak menjalankan model AI dan tidak mengubah anotasi.</p>
        </Card>
      </div>
    </div>
  );
};
