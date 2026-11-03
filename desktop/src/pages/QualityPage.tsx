import React, { useEffect, useState } from 'react';
import type { DatasetStats, ImageMeta, Page } from '../types';
import { fetchActiveLearningQueue, fetchDatasetStats, type ReviewQueueItem } from '../services/api';
import { Badge, Button, Card, Icon, Note, PageHeader, ProgressBar } from '../components/ui';
import { formatCount } from '../format';
import { firstPendingIndex, summarizeDataset } from './datasetSummary';
import { getClassColor } from '../types';

interface Props {
  images: ImageMeta[];
  onNavigate: (page: Page) => void;
  onOpenImage: (index: number) => void;
}

export const QualityPage: React.FC<Props> = ({ images, onNavigate, onOpenImage }) => {
  const summary = summarizeDataset(images);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [queue, setQueue] = useState<ReviewQueueItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    Promise.all([fetchDatasetStats(), fetchActiveLearningQueue(6)])
      .then(([nextStats, nextQueue]) => {
        if (!active) return;
        setStats(nextStats);
        setQueue(nextQueue.queue.filter(item => item.difficulty > 0));
      })
      .catch(err => { if (active) setError(String(err)); });
    return () => { active = false; };
  }, [images.length]); // approvals change counts the page already derives from the image list

  const counts = Object.entries(stats?.class_counts || {}).sort((a, b) => b[1] - a[1]);
  const totalLabels = counts.reduce((sum, [, count]) => sum + count, 0);
  const maxCount = Math.max(1, ...counts.map(([, count]) => count));
  const openByPath = (path: string) => {
    const index = images.findIndex(image => image.path === path);
    if (index >= 0) onOpenImage(index);
  };

  const tiles = [
    { label: 'Gambar disetujui', value: summary.approved, note: `${summary.approvedPercent}% dari dataset` },
    { label: 'Menunggu review', value: summary.pending, note: 'belum ikut ekspor yang disetujui' },
    { label: 'Hasil auto-label', value: summary.autoLabeled, note: 'belum dikonfirmasi orang' },
    { label: 'Anotasi total', value: summary.annotations, note: 'kotak dan poligon di dataset' },
  ];

  return (
    <div className="mx-auto max-w-[1040px] px-8 py-10">
      <PageHeader
        title="Periksa kesiapan dataset"
        subtitle="Periksa review manusia dan konsistensi kelas sebelum mengekspor."
        actions={<Button onClick={() => onNavigate('export')}>Ekspor yang disetujui</Button>}
      />

      <Card className="grid grid-cols-2 divide-line lg:grid-cols-4 lg:divide-x">
        {tiles.map(tile => (
          <div key={tile.label} className="p-6">
            <div className="text-xs text-ink-500">{tile.label}</div>
            <div className="mt-3 text-[28px] font-bold leading-none tabular-nums text-ink-900">{formatCount(tile.value)}</div>
            <div className="mt-3 text-[11px] text-ink-500">{tile.note}</div>
          </div>
        ))}
      </Card>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-[1fr_352px]">
        <Card className="p-6">
          <h2 className="text-lg font-bold text-ink-900">Prioritas review</h2>
          <div className="mt-4 divide-y divide-line">
            <div className="flex items-start gap-3 py-4">
              <Icon name="warning" className="text-[20px] text-ink-500" />
              <div className="min-w-0 flex-1">
                <div className="text-[13px] font-semibold text-ink-900">Label belum dikonfirmasi</div>
                <div className="mt-0.5 text-xs text-ink-500">Hasil anotasi perlu persetujuan reviewer sebelum ikut ekspor.</div>
              </div>
              <Badge tone={summary.pending ? 'warn' : 'ok'}>{formatCount(summary.pending)} gambar</Badge>
            </div>
            <div className="flex items-start gap-3 py-4">
              <Icon name="center_focus_weak" className="text-[20px] text-ink-500" />
              <div className="min-w-0 flex-1">
                <div className="text-[13px] font-semibold text-ink-900">Gambar paling sulit</div>
                <div className="mt-0.5 text-xs text-ink-500">
                  Diurutkan dari skor kesulitan (kepadatan, objek kecil, tumpang tindih). Gunakan <b>Cek kualitas</b> di review untuk mencari objek yang terlewat.
                </div>
                {queue.length > 0 ? (
                  <ul className="mt-3 space-y-1.5">
                    {queue.map(item => (
                      <li key={item.path}>
                        <button type="button" onClick={() => openByPath(item.path)}
                          className="flex w-full items-center gap-3 rounded-md px-2 py-1.5 text-left text-xs hover:bg-brand-50 cursor-pointer">
                          <span className="min-w-0 flex-1 truncate text-ink-700">{item.filename}</span>
                          <span className="tabular-nums text-ink-500">{Math.round(item.difficulty * 100)}</span>
                          <Icon name="arrow_forward" className="text-[14px] text-ink-400" />
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-2 text-xs text-ink-400">Skor muncul setelah gambar disimpan atau disetujui.</p>
                )}
              </div>
            </div>
          </div>
          <Button block className="mt-2" disabled={!images.length} onClick={() => onOpenImage(firstPendingIndex(images))}>
            Buka antrean review
          </Button>
          <Note className="mt-5" title="Persetujuan manusia menentukan ekspor">
            Angka ini menunjukkan kelengkapan review, bukan ukuran akurasi model.
          </Note>
        </Card>

        <Card className="h-fit p-6">
          <h2 className="text-lg font-bold text-ink-900">Distribusi label</h2>
          <div className="mt-2 text-xs text-ink-500">{formatCount(totalLabels)} objek · {counts.length} kelas</div>
          {error && <p role="alert" className="mt-3 text-xs text-danger-600">Statistik tidak bisa dimuat: {error}</p>}
          <div className="mt-5 space-y-4">
            {counts.map(([name, count]) => (
              <div key={name}>
                <div className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2 text-ink-700">
                    <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: getClassColor(name).stroke }} />
                    {name}
                  </span>
                  <span className="tabular-nums text-ink-900">{formatCount(count)}</span>
                </div>
                <ProgressBar value={(count / maxCount) * 100} className="mt-2" />
              </div>
            ))}
            {!counts.length && !error && <p className="text-xs text-ink-400">{stats ? 'Belum ada label.' : 'Memuat…'}</p>}
          </div>
        </Card>
      </div>

      <Note className="mt-6" title={`${formatCount(summary.approved)} gambar dapat diekspor`}>
        Mode ekspor “disetujui” hanya menyertakan gambar yang sudah disetujui. Gambar lain tetap di antrean review.
      </Note>
    </div>
  );
};
