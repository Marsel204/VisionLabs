import React, { useEffect, useState } from 'react';
import type { BoundingBox, ImageMeta, Page, SystemHealth } from '../types';
import { fetchAnnotations, getImageUrl, imageIdentity } from '../services/api';
import { Badge, Button, Card, Icon, PageHeader, ProgressBar, StatRow } from '../components/ui';
import { formatCount } from '../format';
import { ImageWithBoxes } from '../components/ImageWithBoxes';
import { firstPendingIndex, summarizeDataset } from './datasetSummary';
import { useAnnotationSources } from './useAnnotationSources';

interface Props {
  images: ImageMeta[];
  health: SystemHealth | null;
  datasetName: string;
  onNavigate: (page: Page) => void;
  onOpenImage: (index: number) => void;
}

export const OverviewPage: React.FC<Props> = ({ images, health, datasetName, onNavigate, onOpenImage }) => {
  const summary = summarizeDataset(images);
  const nextIndex = firstPendingIndex(images);
  const next = images[nextIndex] || null;
  const [preview, setPreview] = useState<BoundingBox[]>([]);
  const nextKey = next ? imageIdentity(next) : '';

  useEffect(() => {
    let active = true;
    setPreview([]);
    if (!next) return;
    fetchAnnotations(nextKey, next.dataset_id)
      .then(data => { if (active) setPreview(data.boxes || []); })
      .catch(() => { /* The preview is optional; the review page reports load errors. */ });
    return () => { active = false; };
  }, [nextKey, next?.dataset_id]); // eslint-disable-line react-hooks/exhaustive-deps

  const { sources } = useAnnotationSources(summary.total > 0 && summary.annotations === 0 ? health?.dataset_dir : undefined);
  const importable = sources || [];
  const importableBoxes = importable.reduce((sum, source) => sum + source.annotations, 0);
  const classCount = health?.classes?.length || 0;
  const status = summary.total === 0 ? null : summary.pending === 0
    ? <Badge tone="ok">Siap ekspor</Badge> : <Badge tone="warn">Perlu review</Badge>;

  return (
    <div className="mx-auto max-w-[1040px] px-8 py-10">
      <PageHeader
        title="Gambar hari ini. Dataset berikutnya."
        subtitle="Lanjutkan anotasi, periksa hasilnya, lalu bawa dataset ke eksperimenmu."
        actions={<Button icon="add" onClick={() => onNavigate('import')}>Impor dataset</Button>}
      />

      {summary.total > 0 && summary.annotations === 0 && importable.length > 0 && (
        <Card tone="soft" className="mb-6 flex flex-wrap items-center gap-4 p-5">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white text-brand-600">
            <Icon name="download" className="text-[20px]" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-semibold text-ink-900">Anotasi dataset ini belum dimuat</div>
            <div className="mt-0.5 text-xs text-ink-500">
              Ditemukan {importable.length} file anotasi COCO dengan {formatCount(importableBoxes)} kotak.
              Impor agar kotaknya muncul di review, audit, dan ekspor.
            </div>
          </div>
          <Button onClick={() => onNavigate('import')}>Impor anotasi</Button>
        </Card>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_332px]">
        <section>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-[15px] font-semibold text-ink-900">Lanjutkan pekerjaan</h2>
            <Badge>Proyek aktif</Badge>
          </div>
          {next ? (
            <button
              type="button"
              onClick={() => onOpenImage(nextIndex)}
              className="group relative block w-full overflow-hidden rounded-xl border border-line bg-canvas text-left cursor-pointer"
              title="Buka gambar ini di review"
            >
              <ImageWithBoxes
                src={getImageUrl(nextKey)}
                alt={next.filename}
                width={next.width}
                height={next.height}
                boxes={preview}
                className="max-h-[360px] overflow-hidden"
                imgClassName="max-h-[360px]"
              />
              <span className="absolute left-3 top-3 flex items-center gap-1.5 rounded-md bg-white/95 px-2 py-1 text-[11px] font-medium text-ink-700 shadow-sm">
                <Icon name="crop_free" className="text-[14px]" /> Bounding box
              </span>
            </button>
          ) : (
            <Card className="flex h-[300px] flex-col items-center justify-center gap-3 border-dashed text-center">
              <Icon name="add_photo_alternate" className="text-[32px] text-ink-400" />
              <p className="text-sm text-ink-500">Belum ada gambar di dataset ini.</p>
              <Button variant="secondary" onClick={() => onNavigate('import')}>Impor dataset</Button>
            </Card>
          )}
          <div className="mt-5 flex items-end justify-between gap-4">
            <div className="min-w-0">
              <div className="truncate text-xl font-bold text-ink-900">{datasetName || 'Dataset'}</div>
              <div className="mt-1 text-[13px] text-ink-500">
                {classCount} kelas objek <span className="mx-2 text-ink-300">/</span> {formatCount(summary.annotations)} anotasi
              </div>
            </div>
            <Button disabled={!next} onClick={() => onOpenImage(nextIndex)}>Lanjutkan review</Button>
          </div>
        </section>

        <Card tone="soft" className="flex flex-col p-5">
          <div className="flex items-center justify-between">
            <h2 className="text-[15px] font-semibold text-ink-900">Kesiapan dataset</h2>
            <Icon name="verified_user" className="text-[20px] text-ink-500" />
          </div>
          <div className="mt-5 flex items-baseline gap-3">
            <span className="text-[40px] font-bold leading-none tracking-tight text-ink-900">{summary.approvedPercent}%</span>
            <span className="text-xs text-ink-500">selesai direview</span>
          </div>
          <ProgressBar value={summary.approvedPercent} className="mt-4" />
          <div className="mt-4">
            <StatRow dot="#2f6fe4" label="Disetujui" value={formatCount(summary.approved)} />
            <StatRow dot="#d9822b" label="Menunggu review" value={formatCount(summary.pending)} />
            <StatRow dot="#b4bfce" label="Total gambar" value={formatCount(summary.total)} />
          </div>
          <div className="mt-auto border-t border-brand-200 pt-5">
            <div className="text-sm font-semibold text-ink-900">Satu langkah sebelum ekspor.</div>
            <p className="mt-1.5 text-xs leading-relaxed text-ink-500">
              Periksa kotak dan kelas objek, lalu setujui gambar yang sudah benar.
            </p>
            <Button variant="secondary" block className="mt-4" disabled={!next} onClick={() => onOpenImage(nextIndex)}>
              Buka antrean review
            </Button>
          </div>
        </Card>
      </div>

      <div className="mt-8 grid grid-cols-1 border-y border-line md:grid-cols-2">
        {[
          { icon: 'center_focus_weak', title: 'Mulai dengan auto-label', text: 'Pilih model dan cek pratinjau sebelum batch.', page: 'autolabel' as Page },
          { icon: 'download', title: 'Bawa dataset ke eksperimen', text: 'YOLO atau COCO, dengan pembagian train/val/test.', page: 'export' as Page },
        ].map((action, index) => (
          <button
            key={action.title}
            type="button"
            onClick={() => onNavigate(action.page)}
            className={`flex items-center gap-4 px-0 py-5 text-left transition-colors hover:bg-brand-50 cursor-pointer md:px-4 ${
              index === 1 ? 'md:border-l md:border-line' : ''
            }`}
          >
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-brand-100 text-brand-600">
              <Icon name={action.icon} className="text-[20px]" />
            </span>
            <span className="min-w-0 flex-1">
              <span className="block text-sm font-semibold text-ink-900">{action.title}</span>
              <span className="mt-0.5 block text-xs text-ink-500">{action.text}</span>
            </span>
            <Icon name="arrow_forward" className="text-[18px] text-ink-400" />
          </button>
        ))}
      </div>

      <section className="mt-8">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-[15px] font-semibold text-ink-900">Proyek kamu</h2>
          <button type="button" onClick={() => onNavigate('projects')}
            className="flex items-center gap-1 text-[13px] text-ink-500 hover:text-brand-600 cursor-pointer">
            Lihat semua proyek <Icon name="arrow_outward" className="text-[16px]" />
          </button>
        </div>
        <div className="overflow-hidden rounded-xl border border-line">
          <table className="w-full text-left text-sm">
            <thead className="bg-brand-50 text-xs text-ink-500">
              <tr>
                <th className="px-4 py-3 font-medium">Nama proyek</th>
                <th className="px-4 py-3 font-medium">Gambar</th>
                <th className="px-4 py-3 font-medium">Progres review</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-line">
                <td className="px-4 py-3">
                  <div className="flex items-center gap-3">
                    {images[0] ? (
                      <img src={getImageUrl(imageIdentity(images[0]))} alt="" className="h-9 w-12 rounded object-cover" />
                    ) : <span className="h-9 w-12 rounded bg-brand-100" />}
                    <div className="min-w-0">
                      <div className="truncate font-semibold text-ink-900">{datasetName || '—'}</div>
                      <div className="text-xs text-ink-500">Deteksi objek</div>
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3 tabular-nums text-ink-700">{formatCount(summary.total)}</td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-3">
                    <ProgressBar value={summary.approvedPercent} className="w-24" />
                    <span className="text-xs tabular-nums text-ink-500">{summary.approvedPercent}%</span>
                  </div>
                </td>
                <td className="px-4 py-3">{status}</td>
                <td className="px-4 py-3 text-right">
                  <button type="button" onClick={() => onNavigate('review')}
                    className="text-[13px] font-semibold text-ink-900 hover:text-brand-600 cursor-pointer">
                    Buka proyek
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-4 flex items-center gap-2 text-[11px] text-ink-500">
          <Icon name="verified_user" className="text-[14px]" />
          Gambar dan koreksi tetap di perangkatmu, kecuali kamu memilih inference cloud atau Colab.
        </p>
      </section>
    </div>
  );
};
