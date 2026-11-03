import React from 'react';
import type { Page } from '../types';

const STEPS: { page: Page; label: string }[] = [
  { page: 'import', label: 'Impor dataset' },
  { page: 'autolabel', label: 'Auto-label' },
  { page: 'review', label: 'Review' },
  { page: 'quality', label: 'Kualitas' },
  { page: 'export', label: 'Ekspor' },
];

/** The five-step dataset workflow shown at the top of guided pages. */
export const WorkflowSteps: React.FC<{ current: Page; onNavigate: (page: Page) => void }> = ({ current, onNavigate }) => (
  <ol className="mb-7 grid grid-cols-5 border-b border-line">
    {STEPS.map((step, index) => {
      const active = step.page === current;
      return (
        <li key={step.page}>
          <button
            type="button"
            onClick={() => onNavigate(step.page)}
            aria-current={active ? 'step' : undefined}
            className={`-mb-px flex w-full items-center gap-2.5 border-b-2 pb-3 text-left cursor-pointer ${
              active ? 'border-brand-600' : 'border-transparent hover:border-line-strong'
            }`}
          >
            <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold ${
              active ? 'bg-brand-100 text-brand-700' : 'bg-[#eef1f6] text-ink-500'
            }`}>
              {String(index + 1).padStart(2, '0')}
            </span>
            <span className="min-w-0">
              <span className={`block truncate text-xs font-semibold ${active ? 'text-brand-600' : 'text-ink-500'}`}>{step.label}</span>
              <span className="block truncate text-[10px] text-ink-400">{active ? 'Langkah saat ini' : 'Langkah berikutnya'}</span>
            </span>
          </button>
        </li>
      );
    })}
  </ol>
);
