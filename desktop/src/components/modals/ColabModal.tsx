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
        if (active) setError('Cannot reach VisionLab. Check that the local app is running.');
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
    catch (failure) { setError(failure instanceof Error ? failure.message : 'The operation failed. Try again.'); }
    finally { actionInProgress.current = false; setPending(false); }
  };

  if (!isOpen) return null;
  const connected = status?.auth_state === 'connected';
  const busy = pending || !!status?.busy;
  const needsCode = status?.auth_state === 'awaiting_code';
  const job = status?.job;
  const button = 'min-h-10 px-4 rounded-lg border border-[#496171] text-sm font-medium text-[#e0e8f0] hover:bg-[#223047] disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer';
  const input = 'w-full min-h-10 rounded-lg border border-[#496171] bg-[#0a0e1a] px-3 text-sm text-[#e0e8f0] focus:outline-2 focus:outline-[#7dd3fc] disabled:opacity-60';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div ref={dialog} role="dialog" aria-modal="true" aria-labelledby="colab-title" tabIndex={-1}
        className="w-full max-w-xl max-h-[90vh] overflow-y-auto rounded-xl border border-[#496171] bg-[#0f1524] p-6 shadow-2xl text-[#e0e8f0]"
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
            <h2 id="colab-title" className="text-xl font-semibold">Annotate with Google Colab</h2>
            <p className="text-sm text-[#bcc9d7] mt-2 leading-relaxed">Use your Colab compute units. VisionLab handles setup, image transfer, and saving the labels.</p>
          </div>
          <button className="p-2 cursor-pointer rounded hover:bg-[#223047]" type="button" aria-label="Close Colab" onClick={onClose}>
            <span className="material-symbols-outlined" aria-hidden="true">close</span>
          </button>
        </div>

        {!status && <p role="status" className="text-sm text-[#bcc9d7]">Checking the connection…</p>}
        {status && !status.supported && <div className="space-y-4">
          <p className="text-sm leading-relaxed">Direct sign-in is available on Linux and macOS. Use the Colab notebook for this installation.</p>
          <div className="flex flex-wrap gap-3">
            <button type="button" className={button} disabled={pending} onClick={() => { void perform(() => downloadColabFile('notebook')); }}>Download notebook</button>
            <button type="button" className={button} disabled={pending} onClick={() => { void perform(() => downloadColabFile('source')); }}>Download source bundle</button>
          </div>
        </div>}

        {status?.supported && <>
          <div className="flex items-center justify-between gap-3 py-3 border-y border-[#496171]">
            <span className="text-sm font-medium">{connected ? 'Google account connected' : needsCode ? 'Finish Google sign-in' : status.auth_state === 'connecting' ? 'Preparing your connection…' : 'Connect your Google account'}</span>
            {connected ? <div className="flex flex-wrap justify-end gap-2">
              <button type="button" className={button} disabled={busy || job?.cleanup_required} onClick={() => { setCode(''); void perform(switchColabAccount); }}>Switch account</button>
              <button type="button" className={button} disabled={busy || job?.cleanup_required} onClick={() => { void perform(disconnectColab); }}>Disconnect</button>
            </div>
              : !busy && <button type="button" className={`${button} bg-[#0e4d6e] hover:bg-[#145f84]`} onClick={() => { void perform(connectColab); }}>Sign in with Google</button>}
          </div>

          {connected && <div className="mt-4 text-sm space-y-2">
            {status.account ? <>
              <p className="break-all font-medium">{status.account.email || 'Account email unavailable'}</p>
              <p className="text-[#bcc9d7]">{status.account.compute_units.toFixed(2)} compute units · {status.account.active_runtimes} active runtime{status.account.active_runtimes === 1 ? '' : 's'}</p>
              {status.account.compute_units <= 0 && <p className="text-[#f8dca0] leading-relaxed">This account has no paid compute units. If your credits are on another Google account, choose Switch account.</p>}
            </> : <p className="text-[#bcc9d7]">Check the connected account before starting a run.</p>}
            <button type="button" className="text-[#7dd3fc] underline cursor-pointer disabled:opacity-50" disabled={busy}
              onClick={() => { void perform(refreshColabAccount); }}>Refresh account details</button>
          </div>}

          {needsCode && <form className="mt-5 space-y-3" onSubmit={event => {
            event.preventDefault(); const value = code; setCode('');
            void perform(() => authorizeColab(value));
          }}>
            <p className="text-sm leading-relaxed text-[#bcc9d7]">Sign in in the browser, then paste the authorization code Google shows. This one-time step connects your account without sharing your password.</p>
            {status.auth_url && <a href={status.auth_url} target="_blank" rel="noopener noreferrer" className="text-sm text-[#7dd3fc] underline">Open Google sign-in</a>}
            <label className="block text-sm" htmlFor="colab-code">Google authorization code</label>
            <input id="colab-code" className={input} type="password" autoComplete="off" spellCheck={false}
              maxLength={4096} value={code} onChange={event => setCode(event.target.value)} />
            <button type="submit" className={button} disabled={!code.trim() || pending}>Finish connection</button>
          </form>}

          {connected && <form className="mt-5 space-y-4" onSubmit={event => {
            event.preventDefault();
            if (health) void perform(() => startColabJob({ dataset_id: health.dataset_dir,
              classes: classes.split(',').map(name => name.trim()).filter(Boolean),
              pipeline, limit, gpu, confidence, only_unannotated: onlyUnannotated }));
          }}>
            <fieldset disabled={busy || !!job?.cleanup_required} className="space-y-4">
              <label className="block text-sm space-y-2"><span>Classes to find</span>
                <input className={input} required value={classes} onChange={event => setClasses(event.target.value)} />
                <span className="block text-xs text-[#bcc9d7]">Separate class names with commas.</span>
              </label>
              <div className="grid grid-cols-2 gap-4">
                <label className="block text-sm space-y-2"><span>Images this run</span>
                  <input className={input} type="number" required min={1} max={100} value={limit} onChange={event => setLimit(Number(event.target.value))} />
                </label>
                <label className="block text-sm space-y-2"><span>Output</span>
                  <select className={input} value={pipeline} onChange={event => setPipeline(event.target.value)}>
                    <option value="dino-masks">Boxes and masks</option><option value="dino-boxes">Boxes only</option>
                    <option value="yolo-boxes">YOLO boxes</option><option value="yolo-masks">YOLO boxes and masks</option>
                    <option value="locate-boxes">Locate Anything 3B boxes</option>
                    <option value="locate-masks">Locate Anything 3B boxes and masks</option>
                  </select>
                </label>
              </div>
              <details className="text-sm">
                <summary className="cursor-pointer text-[#7dd3fc] min-h-8">More options</summary>
                <div className="grid grid-cols-2 gap-4 mt-2 mb-4">
                  <label className="block space-y-2"><span>Colab GPU</span>
                    <select className={input} value={gpu} onChange={event => setGpu(event.target.value)}>
                      {(status.supported_gpus || ['T4', 'L4']).map(name => <option key={name} value={name}>{name}</option>)}
                    </select>
                  </label>
                  <label className="block space-y-2"><span>Minimum confidence</span>
                    <input className={input} type="number" required min={0} max={1} step={0.05} value={confidence} onChange={event => setConfidence(Number(event.target.value))} />
                  </label>
                </div>
                <p className="text-xs leading-relaxed text-[#bcc9d7] mb-4">GPU access depends on your Colab account and current availability, even with compute units.</p>
                <label className="flex items-center gap-3"><input type="checkbox" checked={onlyUnannotated} onChange={event => setOnlyUnannotated(event.target.checked)} />Skip images that already have labels</label>
              </details>
              <p className="text-xs leading-relaxed text-[#bcc9d7]">Runs on the next matching images in this dataset. Starting uses your Colab compute units. VisionLab releases the GPU when the job finishes.</p>
              <button type="submit" disabled={!health || !classes.trim()} className={`${button} bg-[#0e4d6e] hover:bg-[#145f84] w-full`}>Start annotation</button>
            </fieldset>
          </form>}

          {job && <div role="status" aria-live="polite" className="mt-5 border-t border-[#496171] pt-4 space-y-3 text-sm">
            <p>{job.stage}</p>
            {status.busy && job.state === 'running' && <progress className="w-full h-2 accent-[#7dd3fc]" max={job.total || 1} value={job.processed || undefined} aria-label="Images processed" />}
            {job.total !== undefined && <p className="text-[#bcc9d7]">{job.saved || 0} saved · {job.failed || 0} failed · {job.conflicts || 0} changed images preserved</p>}
          </div>}

          {job?.cleanup_required && <div role="alert" className="mt-4 space-y-3 text-sm text-[#f8dca0]">
            <p>The GPU may still be running. Release it here or stop the runtime in Google Colab.</p>
            <div className="flex flex-wrap gap-3">
              <button type="button" className={button} disabled={busy} onClick={() => { void perform(stopColab); }}>Release GPU</button>
              <button type="button" className={button} disabled={busy} onClick={() => { void perform(connectColab); }}>Reconnect</button>
              <a className="text-[#7dd3fc] underline self-center" href="https://colab.research.google.com/" target="_blank" rel="noopener noreferrer">Open Colab</a>
            </div>
            <button type="button" className="text-[#7dd3fc] underline cursor-pointer disabled:opacity-50" disabled={busy}
              onClick={() => { void perform(acknowledgeColabRelease); }}>I've stopped it in Colab</button>
          </div>}
          {status.busy && <button type="button" className={`${button} mt-4`} disabled={pending} onClick={() => { void perform(stopColab); }}>{job?.state === 'running' ? 'Stop annotation' : connected ? 'Cancel account check' : 'Cancel sign-in'}</button>}
        </>}
        {(error || status?.error) && <p role="alert" className="mt-4 text-sm text-[#ffb4b4] leading-relaxed">{error || status?.error}</p>}
      </div>
    </div>
  );
};
