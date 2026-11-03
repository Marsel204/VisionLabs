import React from 'react';
import type { AuditFinding, AuditReport } from '../../types';
import { AUDIT_DEPTHS, AUDIT_KIND_LABELS, describeFinding } from '../../audit';
import { Icon } from '../ui';

interface Props {
  report: AuditReport | null;
  findings: AuditFinding[];
  selectedFindingId: string | null;
  isRunning: boolean;
  unflaggedOcclusionCount: number;
  depth: string;
  onChangeDepth: (depth: string) => void;
  onRun: () => void;
  onClose: () => void;
  onSelect: (id: string) => void;
  onApply: (finding: AuditFinding) => void;
  onDismiss: (finding: AuditFinding) => void;
  onFlagOccluded: () => void;
}

/** Audit findings for the current image, shown as a tab of the annotation panel. */
export const AuditPanel: React.FC<Props> = ({
  report, findings, selectedFindingId, isRunning, unflaggedOcclusionCount,
  depth, onChangeDepth, onRun, onClose, onSelect, onApply, onDismiss, onFlagOccluded,
}) => (
  <>
    <div className="shrink-0 space-y-2 border-b border-line p-3">
      <div className="flex gap-1.5">
        <select value={depth} onChange={e => onChangeDepth(e.target.value)} disabled={isRunning}
          aria-label="Kedalaman audit"
          title="Jumlah lapisan deteksi. Jalankan ulang setelah mengubah."
          className="h-8 min-w-0 flex-1 rounded-md border border-line-strong bg-white px-2 text-[11px] text-ink-700 cursor-pointer focus:outline-none focus:border-brand-500">
          {AUDIT_DEPTHS.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
        </select>
        <button type="button" onClick={onRun} disabled={isRunning} title="Jalankan ulang audit (Q)" aria-label="Jalankan ulang audit"
          className="flex h-8 w-8 items-center justify-center rounded-md border border-line-strong text-ink-500 hover:bg-brand-50 hover:text-brand-600 cursor-pointer disabled:opacity-50">
          <Icon name={isRunning ? 'progress_activity' : 'refresh'} className={`text-[16px] ${isRunning ? 'animate-spin' : ''}`} />
        </button>
        <button type="button" onClick={onClose} title="Tutup audit" aria-label="Tutup audit"
          className="flex h-8 w-8 items-center justify-center rounded-md border border-line-strong text-ink-500 hover:bg-brand-50 cursor-pointer">
          <Icon name="close" className="text-[16px]" />
        </button>
      </div>
      {report && (
        <div className="flex flex-wrap gap-1">
          {Object.entries(AUDIT_KIND_LABELS).map(([kind, meta]) => {
            const count = findings.filter(item => item.kind === kind).length;
            return count > 0 ? (
              <span key={kind} className="rounded px-1.5 py-0.5 text-[10px] font-medium"
                style={{ color: meta.color, backgroundColor: `${meta.color}17` }}>
                {count} {meta.label.toLowerCase()}
              </span>
            ) : null;
          })}
          {findings.length === 0 && (
            <span className="text-[11px] text-ok-600">
              {report.prediction_count === 0 ? 'Detektor tidak menemukan apa pun untuk dibandingkan.' : 'Tidak ada masalah ditemukan.'}
            </span>
          )}
        </div>
      )}
      {unflaggedOcclusionCount > 0 && (
        <button type="button" onClick={onFlagOccluded}
          className="w-full rounded-md bg-brand-50 py-1.5 text-[11px] font-semibold text-brand-600 hover:bg-brand-100 cursor-pointer">
          Tandai {unflaggedOcclusionCount} kotak tertutup
        </button>
      )}
    </div>

    <div className="min-h-0 flex-1 space-y-1.5 overflow-y-auto p-2">
      {!report && !isRunning && (
        <p className="p-2 text-[11px] leading-relaxed text-ink-500">
          Bandingkan kotak di gambar ini dengan deteksi ambang rendah per ubin untuk menemukan objek yang terlewat (sering
          tertutup) dan kotak yang mencurigakan.
        </p>
      )}
      {isRunning && <p className="p-2 text-[11px] text-brand-600">Memeriksa…</p>}
      {findings.map(finding => {
        const meta = AUDIT_KIND_LABELS[finding.kind];
        const selected = finding.id === selectedFindingId;
        return (
          <div key={finding.id} onClick={() => onSelect(finding.id)}
            className={`cursor-pointer space-y-1.5 rounded-lg border p-2.5 ${selected ? 'bg-brand-50' : 'bg-white hover:bg-brand-50/60'}`}
            style={{ borderColor: selected ? meta.color : '#e4e9f1' }}>
            <div className="flex items-center justify-between gap-2">
              <span className="flex items-center gap-1 text-[11px] font-semibold" style={{ color: meta.color }}>
                <Icon name={meta.icon} className="text-[15px]" />
                {meta.label}
                {finding.likely_occluded && <span className="text-[10px] font-medium text-ink-500">· tertutup</span>}
              </span>
              <span className="text-[10px] tabular-nums text-ink-400" title="Tingkat keparahan">{Math.round(finding.severity * 100)}%</span>
            </div>
            <p className="text-[11px] leading-snug text-ink-700">{describeFinding(finding)}</p>
            {selected && (
              <div className="flex gap-1.5">
                <button type="button" onClick={e => { e.stopPropagation(); onApply(finding); }}
                  className="h-7 flex-1 rounded-md bg-brand-600 text-[11px] font-semibold text-white hover:bg-brand-700 cursor-pointer">
                  {meta.action} [Y]
                </button>
                <button type="button" onClick={e => { e.stopPropagation(); onDismiss(finding); }}
                  className="h-7 flex-1 rounded-md border border-line-strong text-[11px] font-semibold text-ink-700 hover:bg-white cursor-pointer">
                  Abaikan [N]
                </button>
              </div>
            )}
          </div>
        );
      })}
    </div>
  </>
);
