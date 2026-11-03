import React from 'react';
import type { Page, SystemHealth } from '../../types';
import { Icon } from '../ui';
import { Logo } from './Logo';

interface Props {
  page: Page;
  onNavigate: (page: Page) => void;
  health: SystemHealth | null;
  datasetName: string;
  reviewCount: number;
}

const NAV: { page: Page; label: string; icon: string }[] = [
  { page: 'overview', label: 'Ringkasan', icon: 'space_dashboard' },
  { page: 'projects', label: 'Proyek', icon: 'folder' },
  { page: 'review', label: 'Review anotasi', icon: 'center_focus_weak' },
  { page: 'autolabel', label: 'Auto-label', icon: 'auto_awesome' },
  { page: 'quality', label: 'Kualitas dataset', icon: 'verified_user' },
  { page: 'export', label: 'Ekspor', icon: 'download' },
];

export const Sidebar: React.FC<Props> = ({ page, onNavigate, health, datasetName, reviewCount }) => {
  const usesCloud = health?.inference?.backend === 'cloud';
  const engine = usesCloud ? 'Inference cloud' : health?.gpu?.available ? health.gpu.device : health ? 'CPU lokal' : 'Menghubungkan…';
  const initial = (datasetName || 'A').charAt(0).toUpperCase();
  return (
    <aside className="flex w-[216px] shrink-0 flex-col border-r border-line bg-brand-50 px-3 pb-4 pt-6">
      <Logo className="px-2" />

      <button
        type="button"
        onClick={() => onNavigate('projects')}
        title="Ganti folder dataset"
        className="mt-8 flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left hover:bg-brand-100 cursor-pointer"
      >
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-line-strong bg-white text-xs font-bold text-ink-900">
          {initial}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[13px] font-semibold text-ink-900">{datasetName || 'Belum ada dataset'}</span>
          <span className="block truncate text-[11px] text-ink-500">Workspace lokal</span>
        </span>
        <Icon name="expand_more" className="text-[18px] text-ink-500" />
      </button>

      <div className="my-4 h-px bg-line-strong/70" />
      <div className="px-2 pb-2 text-[11px] font-medium text-ink-500">Workspace</div>
      <nav className="flex flex-col gap-1" aria-label="Navigasi utama">
        {NAV.map(item => {
          const active = item.page === page;
          return (
            <button
              key={item.page}
              type="button"
              onClick={() => onNavigate(item.page)}
              aria-current={active ? 'page' : undefined}
              className={`flex h-10 items-center gap-2.5 whitespace-nowrap rounded-lg px-2.5 text-sm font-medium transition-colors cursor-pointer ${
                active ? 'bg-brand-600 text-white shadow-sm' : 'text-ink-700 hover:bg-brand-100 hover:text-ink-900'
              }`}
            >
              <Icon name={item.icon} className="text-[19px]" />
              <span className="min-w-0 flex-1 truncate text-left">{item.label}</span>
              {item.page === 'review' && reviewCount > 0 && (
                <span className={`rounded-md px-1.5 py-0.5 text-[11px] font-semibold tabular-nums ${
                  active ? 'bg-white/20 text-white' : 'bg-brand-100 text-brand-700'
                }`}>
                  {reviewCount > 999 ? '999+' : reviewCount}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="my-4 h-px bg-line-strong/70" />
      <div className="px-2 pb-2 text-[11px] font-medium text-ink-500">Proyek aktif</div>
      <button
        type="button"
        onClick={() => onNavigate('review')}
        className="flex items-center gap-2 rounded-md px-2 py-1 text-left text-xs text-ink-700 hover:text-brand-600 cursor-pointer"
      >
        <span className="h-2 w-2 shrink-0 rounded-sm border border-ink-400" />
        <span className="truncate">{datasetName || '—'}</span>
      </button>

      <div className="mt-auto space-y-4">
        <div className="flex items-center gap-3 rounded-lg px-2.5 py-2 text-sm text-ink-700" title="Mesin AI yang dipakai untuk deteksi">
          <Icon name={usesCloud ? 'cloud' : 'memory'} className="text-[20px] text-ink-500" />
          <span className="min-w-0 flex-1">
            <span className="block text-[13px] font-medium">Mesin AI</span>
            <span className="block truncate text-[11px] text-ink-500">{engine}</span>
          </span>
        </div>
        <div className="flex gap-2.5 px-2">
          <Icon name="verified_user" className="text-[18px] text-ink-500" />
          <div className="text-[11px] leading-snug">
            <div className="font-semibold text-ink-900">Gambar tetap privat</div>
            <div className="text-ink-500">{usesCloud ? 'Inference lewat cloud-mu.' : 'Tersimpan di perangkatmu.'}</div>
          </div>
        </div>
      </div>
    </aside>
  );
};
