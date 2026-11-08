import React from 'react';
import { useAppUpdate } from '../../hooks/useAppUpdate';
import { Icon } from '../ui';

const BASE =
  'flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-semibold cursor-pointer ' +
  'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600';

/** Version badge and one-click updater for the desktop app. Renders nothing in a browser. */
export const UpdateIndicator: React.FC = () => {
  const { state, check, download, install } = useAppUpdate();
  if (!state) return null;

  const version = `v${state.currentVersion}`;
  const notes = state.releaseNotes ? `\n\n${state.releaseNotes}` : '';

  switch (state.status) {
    case 'available':
      return (
        <button
          type="button"
          onClick={download}
          title={`Unduh versi ${state.availableVersion}${notes}`}
          className={`${BASE} bg-brand-50 text-brand-600 hover:bg-brand-100`}
        >
          <Icon name="download" className="text-[18px]" />
          Pembaruan v{state.availableVersion}
        </button>
      );
    case 'downloading':
      return (
        <div
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={state.percent}
          aria-label="Mengunduh pembaruan"
          className="flex items-center gap-2 px-2 py-1 text-xs font-semibold text-brand-600"
        >
          <span>Mengunduh {state.percent}%</span>
          <span className="h-1 w-16 overflow-hidden rounded-full bg-brand-100">
            <span className="block h-full bg-brand-600" style={{ width: `${state.percent}%` }} />
          </span>
        </div>
      );
    case 'downloaded':
      return (
        <button
          type="button"
          onClick={install}
          title={`Versi ${state.availableVersion} siap. Aplikasi akan menutup dan terbuka lagi otomatis.`}
          className={`${BASE} bg-brand-600 text-white hover:bg-brand-700`}
        >
          <Icon name="refresh" className="text-[18px]" />
          Mulai ulang untuk memperbarui
        </button>
      );
    case 'checking':
      return (
        <span className="flex items-center gap-1.5 px-2 py-1 text-xs text-ink-500" role="status">
          <Icon name="refresh" className="animate-spin text-[18px]" />
          Memeriksa pembaruan…
        </span>
      );
    case 'error':
      return (
        <button
          type="button"
          onClick={state.availableVersion ? download : check}
          title={`${state.error ?? 'Pembaruan gagal'}\nKlik untuk mencoba lagi.`}
          className={`${BASE} bg-danger-50 text-danger-600 hover:bg-danger-50`}
        >
          <Icon name="warning" className="text-[18px]" />
          Pembaruan gagal · coba lagi
        </button>
      );
    case 'disabled':
      return (
        <span className="px-2 py-1 text-xs text-ink-400" title={state.disabledReason ?? undefined}>
          {version}
        </span>
      );
    default:
      return (
        <button
          type="button"
          onClick={check}
          title="Periksa pembaruan"
          className={`${BASE} font-medium text-ink-500 hover:bg-brand-50 hover:text-brand-600`}
        >
          {state.status === 'up-to-date' && <Icon name="check_circle" className="text-[18px] text-ok-600" />}
          {state.status === 'up-to-date' ? `${version} · terbaru` : version}
        </button>
      );
  }
};
