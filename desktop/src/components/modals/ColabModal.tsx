import React, { useCallback, useEffect, useRef, useState } from 'react';
import type { SystemHealth } from '../../types';
import {
  acknowledgeColabRelease, authorizeColab, connectColab, disconnectColab, downloadColabFile,
  fetchColabStatus, startColabJob, stopColab,
  refreshColabAccount, switchColabAccount,
} from '../../services/api';
import type { ColabStatus } from '../../services/api';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  health: SystemHealth | null;
  onComplete: () => void;
}

export const ColabModal: React.FC<Props> = ({ isOpen, onClose, health, onComplete }) => {
  const [status, setStatus] = useState<ColabStatus | null>(null);
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const [code, setCode] = useState('');
  const [classes, setClasses] = useState('');
  const [limit, setLimit] = useState(3);
  const [pipeline, setPipeline] = useState('dino-masks');
  const [gpu, setGpu] = useState('T4');
  const [onlyUnannotated, setOnlyUnannotated] = useState(true);
  const [confidence, setConfidence] = useState(0.35);
  const dialog = useRef<HTMLDivElement>(null);
  const actionInProgress = useRef(false);
  const pollInProgress = useRef(false);
  const generation = useRef(0);
  const completedJob = useRef('');
  const callback = useRef(onComplete);

  useEffect(() => { callback.current = onComplete; }, [onComplete]);

  const accept = useCallback((next: ColabStatus) => {
    setStatus(next);
    if (!next.busy && next.job?.state === 'completed' && next.job.id
        && completedJob.current !== next.job.id) {
      completedJob.current = next.job.id;
      callback.current();
    }
  }, []);

  useEffect(() => {
    if (!isOpen && !status?.busy) return;
    let active = true;
    const poll = async () => {
      if (pollInProgress.current || actionInProgress.current) return;
      pollInProgress.current = true;
      const ticket = ++generation.current;
      try {
        const next = await fetchColabStatus();
        if (active && ticket === generation.current) { setError(''); accept(next); }
      } catch {
        if (active) setError('Anotara tidak bisa dihubungi. Pastikan aplikasi lokal sedang berjalan.');
      } finally { pollInProgress.current = false; }
    };
    void poll();
    const timer = setInterval(() => { void poll(); }, 1500);
    return () => { active = false; clearInterval(timer); };
  }, [isOpen, status?.busy, accept]);

  useEffect(() => {
    if (isOpen) setClasses(health?.classes.join(', ') || 'object');
    else setCode('');
  }, [isOpen, health?.dataset_dir, health?.classes]);

  useEffect(() => {
    if (!isOpen) return;
    const previous = document.activeElement as HTMLElement | null;
    dialog.current?.focus();
    return () => { previous?.focus(); };
  }, [isOpen]);

  const perform = async (operation: () => Promise<ColabStatus | void>) => {
    actionInProgress.current = true;
    ++generation.current;
    setPending(true); setError('');
    try { const next = await operation(); if (next) accept(next); }
    catch (failure) { setError(failure instanceof Error ? failure.message : 'Operasi gagal. Coba lagi.'); }
    finally { actionInProgress.current = false; setPending(false); }
  };

  if (!isOpen) return null;
  const connected = status?.auth_state === 'connected';
  const busy = pending || !!status?.busy;
  const needsCode = status?.auth_state === 'awaiting_code';
  const job = status?.job;
  const button = 'min-h-10 px-4 rounded-lg border border-line-strong bg-white text-sm font-semibold text-ink-900 hover:bg-brand-50 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer';
  const primary = 'min-h-10 px-4 rounded-lg border border-brand-600 bg-brand-600 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer';
  const input = 'w-full min-h-10 rounded-lg border border-line-strong bg-white px-3 text-sm text-ink-900 focus:outline-none focus:border-brand-500 disabled:opacity-60';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink-900/30 p-4 backdrop-blur-[2px]">
      <div ref={dialog} role="dialog" aria-modal="true" aria-labelledby="colab-title" tabIndex={-1}
        className="w-full max-w-xl max-h-[90vh] overflow-y-auto rounded-2xl border border-line bg-white p-8 shadow-2xl text-ink-900"
        onKeyDown={event => {
          event.stopPropagation();
          if (event.key === 'Escape') onClose();
          if (event.key === 'Tab') {
            const controls = Array.from(dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), a[href]') || []);
            const first = controls[0], last = controls[controls.length - 1];
            if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) {
              event.preventDefault(); last?.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
              event.preventDefault(); first?.focus();
            }
          }
        }}>
        <div className="flex items-start justify-between gap-4 mb-5">
          <div>
            <h2 id="colab-title" className="text-2xl font-bold tracking-tight">Anotasi dengan Google Colab</h2>
            <p className="text-sm text-ink-500 mt-2 leading-relaxed">Pakai unit komputasi Colab-mu. Anotara mengurus persiapan, transfer gambar, dan penyimpanan label.</p>
          </div>
          <button className="p-2 cursor-pointer rounded-md text-ink-500 hover:bg-brand-50" type="button" aria-label="Tutup Colab" onClick={onClose}>
            <span className="material-symbols-outlined" aria-hidden="true">close</span>
          </button>
        </div>

        {!status && <p role="status" className="text-sm text-ink-500">Memeriksa koneksi…</p>}
        {status && !status.supported && <div className="space-y-4">
          <p className="text-sm leading-relaxed text-ink-700">Masuk langsung tersedia di Linux dan macOS. Untuk instalasi ini, pakai notebook Colab.</p>
          <div className="flex flex-wrap gap-3">
            <button type="button" className={button} disabled={pending} onClick={() => { void perform(() => downloadColabFile('notebook')); }}>Unduh notebook</button>
            <button type="button" className={button} disabled={pending} onClick={() => { void perform(() => downloadColabFile('source')); }}>Unduh paket sumber</button>
          </div>
        </div>}

        {status?.supported && <>
          <div className="flex items-center justify-between gap-3 py-3 border-y border-line">
            <span className="text-sm font-medium">{connected ? 'Akun Google terhubung' : needsCode ? 'Selesaikan masuk Google' : status.auth_state === 'connecting' ? 'Menyiapkan koneksi…' : 'Hubungkan akun Google-mu'}</span>
            {connected ? <div className="flex flex-wrap justify-end gap-2">
              <button type="button" className={button} disabled={busy || job?.cleanup_required} onClick={() => { setCode(''); void perform(switchColabAccount); }}>Ganti akun</button>
              <button type="button" className={button} disabled={busy || job?.cleanup_required} onClick={() => { void perform(disconnectColab); }}>Putuskan</button>
            </div>
              : !busy && <button type="button" className={primary} onClick={() => { void perform(connectColab); }}>Masuk dengan Google</button>}
          </div>

          {connected && <div className="mt-4 text-sm space-y-2">
            {status.account ? <>
              <p className="break-all font-medium">{status.account.email || 'Email akun tidak tersedia'}</p>
              <p className="text-ink-500">{status.account.compute_units.toFixed(2)} unit komputasi · {status.account.active_runtimes} runtime aktif</p>
              {status.account.compute_units <= 0 && <p className="text-warn-700 leading-relaxed">Akun ini tidak punya unit komputasi berbayar. Jika kreditmu ada di akun Google lain, pilih Ganti akun.</p>}
            </> : <p className="text-ink-500">Periksa akun yang terhubung sebelum memulai.</p>}
            <button type="button" className="text-brand-600 font-semibold cursor-pointer disabled:opacity-50" disabled={busy}
              onClick={() => { void perform(refreshColabAccount); }}>Muat ulang detail akun</button>
          </div>}

          {needsCode && <form className="mt-5 space-y-3" onSubmit={event => {
            event.preventDefault(); const value = code; setCode('');
            void perform(() => authorizeColab(value));
          }}>
            <p className="text-sm leading-relaxed text-ink-500">Masuk di browser, lalu tempel kode otorisasi dari Google. Langkah sekali ini menghubungkan akunmu tanpa membagikan kata sandi.</p>
            {status.auth_url && <a href={status.auth_url} target="_blank" rel="noopener noreferrer" className="text-sm font-semibold text-brand-600">Buka halaman masuk Google</a>}
            <label className="block text-sm font-semibold" htmlFor="colab-code">Kode otorisasi Google</label>
            <input id="colab-code" className={input} type="password" autoComplete="off" spellCheck={false}
              maxLength={4096} value={code} onChange={event => setCode(event.target.value)} />
            <button type="submit" className={primary} disabled={!code.trim() || pending}>Selesaikan koneksi</button>
          </form>}

          {connected && <form className="mt-5 space-y-4" onSubmit={event => {
            event.preventDefault();
            if (health) void perform(() => startColabJob({ dataset_id: health.dataset_dir,
              classes: classes.split(',').map(name => name.trim()).filter(Boolean),
              pipeline, limit, gpu, confidence, only_unannotated: onlyUnannotated }));
          }}>
            <fieldset disabled={busy || !!job?.cleanup_required} className="space-y-4">
              <label className="block text-sm space-y-2"><span className="font-semibold">Kelas yang dicari</span>
                <input className={input} required value={classes} onChange={event => setClasses(event.target.value)} />
                <span className="block text-xs text-ink-500">Pisahkan nama kelas dengan koma.</span>
              </label>
              <div className="grid grid-cols-2 gap-4">
                <label className="block text-sm space-y-2"><span className="font-semibold">Gambar per run</span>
                  <input className={input} type="number" required min={1} max={100} value={limit} onChange={event => setLimit(Number(event.target.value))} />
                </label>
                <label className="block text-sm space-y-2"><span className="font-semibold">Hasil</span>
                  <select className={input} value={pipeline} onChange={event => setPipeline(event.target.value)}>
                    <option value="dino-masks">Kotak dan mask</option><option value="dino-boxes">Kotak saja</option>
                    <option value="yolo-boxes">Kotak YOLO</option><option value="yolo-masks">Kotak dan mask YOLO</option>
                    <option value="locate-boxes">Kotak Locate Anything 3B</option>
                    <option value="locate-masks">Kotak dan mask Locate Anything 3B</option>
                  </select>
                </label>
              </div>
              <details className="text-sm">
                <summary className="cursor-pointer font-semibold text-brand-600 min-h-8">Opsi lainnya</summary>
                <div className="grid grid-cols-2 gap-4 mt-2 mb-4">
                  <label className="block space-y-2"><span>Colab GPU</span>
                    <select className={input} value={gpu} onChange={event => setGpu(event.target.value)}>
                      {(status.supported_gpus || ['T4', 'L4']).map(name => <option key={name} value={name}>{name}</option>)}
                    </select>
                  </label>
                  <label className="block space-y-2"><span>Confidence minimum</span>
                    <input className={input} type="number" required min={0} max={1} step={0.05} value={confidence} onChange={event => setConfidence(Number(event.target.value))} />
                  </label>
                </div>
                <p className="text-xs leading-relaxed text-ink-500 mb-4">Akses GPU tergantung akun Colab-mu dan ketersediaan saat ini, meski punya unit komputasi.</p>
                <label className="flex items-center gap-3"><input type="checkbox" className="accent-brand-600" checked={onlyUnannotated} onChange={event => setOnlyUnannotated(event.target.checked)} />Lewati gambar yang sudah punya label</label>
              </details>
              <p className="text-xs leading-relaxed text-ink-500">Berjalan pada gambar berikutnya yang cocok di dataset ini. Memulai memakai unit komputasi Colab-mu. Anotara melepas GPU saat job selesai.</p>
              <button type="submit" disabled={!health || !classes.trim()} className={`${primary} w-full`}>Mulai anotasi</button>
            </fieldset>
          </form>}

          {job && <div role="status" aria-live="polite" className="mt-5 border-t border-line pt-4 space-y-3 text-sm">
            <p>{job.stage}</p>
            {status.busy && job.state === 'running' && <progress className="w-full h-2 accent-brand-600" max={job.total || 1} value={job.processed || undefined} aria-label="Gambar diproses" />}
            {job.total !== undefined && <p className="text-ink-500">{job.saved || 0} tersimpan · {job.failed || 0} gagal · {job.conflicts || 0} gambar yang berubah dipertahankan</p>}
          </div>}

          {job?.cleanup_required && <div role="alert" className="mt-4 space-y-3 rounded-lg bg-warn-50 p-4 text-sm text-warn-700">
            <p>GPU mungkin masih berjalan. Lepaskan di sini atau hentikan runtime di Google Colab.</p>
            <div className="flex flex-wrap gap-3">
              <button type="button" className={button} disabled={busy} onClick={() => { void perform(stopColab); }}>Lepaskan GPU</button>
              <button type="button" className={button} disabled={busy} onClick={() => { void perform(connectColab); }}>Hubungkan ulang</button>
              <a className="font-semibold text-brand-600 self-center" href="https://colab.research.google.com/" target="_blank" rel="noopener noreferrer">Buka Colab</a>
            </div>
            <button type="button" className="font-semibold text-brand-600 cursor-pointer disabled:opacity-50" disabled={busy}
              onClick={() => { void perform(acknowledgeColabRelease); }}>Sudah kuhentikan di Colab</button>
          </div>}
          {status.busy && <button type="button" className={`${button} mt-4`} disabled={pending} onClick={() => { void perform(stopColab); }}>{job?.state === 'running' ? 'Hentikan anotasi' : connected ? 'Batalkan cek akun' : 'Batalkan masuk'}</button>}
        </>}
        {(error || status?.error) && <p role="alert" className="mt-4 text-sm text-danger-600 leading-relaxed">{error || status?.error}</p>}
      </div>
    </div>
  );
};
