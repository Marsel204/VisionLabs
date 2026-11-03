import React from 'react';
import type { SystemHealth } from '../../types';
import { Icon } from '../ui';

interface Props {
  crumb: string;
  health: SystemHealth | null;
  onOpenShortcuts: () => void;
  onOpenColab: () => void;
}

export const Topbar: React.FC<Props> = ({ crumb, health, onOpenShortcuts, onOpenColab }) => {
  const usesCloud = health?.inference?.backend === 'cloud';
  const status = !health ? 'Menghubungkan…' : usesCloud ? 'Inference cloud' : 'Mesin lokal';
  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-line bg-white px-8">
      <nav aria-label="Breadcrumb" className="flex items-center gap-3 text-[13px]">
        <span className="text-ink-500">Workspace</span>
        <span className="text-ink-300">/</span>
        <span className="font-semibold text-ink-900">{crumb}</span>
      </nav>
      <div className="flex items-center gap-4 text-xs">
        <span className="flex items-center gap-1.5 text-ink-500">
          <span className={`h-1.5 w-1.5 rounded-full ${health ? 'bg-ok-600' : 'bg-ink-300'}`} />
          {status}
        </span>
        <span className="h-6 w-px bg-line" />
        <button
          type="button"
          onClick={onOpenColab}
          title="Jalankan anotasi di GPU Google Colab"
          className="flex items-center gap-1.5 rounded-md px-2 py-1 text-ink-700 hover:bg-brand-50 hover:text-brand-600 cursor-pointer"
        >
          <Icon name="cloud" className="text-[18px]" />
          <span className="font-semibold">Colab</span>
        </button>
        <button
          type="button"
          onClick={onOpenShortcuts}
          title="Pintasan keyboard (?)"
          aria-label="Pintasan keyboard"
          className="flex h-8 w-8 items-center justify-center rounded-md text-ink-500 hover:bg-brand-50 hover:text-brand-600 cursor-pointer"
        >
          <Icon name="keyboard" className="text-[18px]" />
        </button>
      </div>
    </header>
  );
};
