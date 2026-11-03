import React, { useState } from 'react';
import type { BoundingBox } from '../../types';
import { getClassColor, LOW_CONFIDENCE, LOW_CONFIDENCE_COLOR } from '../../types';
import { Icon } from '../ui';

interface Props {
  boxes: BoundingBox[];
  selectedBoxId: string | null;
  availableClasses: string[];
  activeClassName: string;
  onSelectBox: (id: string | null) => void;
  onUpdateBox: (box: BoundingBox) => void;
  onDeleteBox: (id: string) => void;
  onAcceptBox: (id: string) => void;
  onChangeActiveClass: (name: string) => void;
  onRegisterClass?: (name: string) => Promise<number>;
  /** When the audit is open, the panel shows a tab for it. */
  audit?: { count: number; content: React.ReactNode } | null;
  tab: 'annotations' | 'audit';
  onTabChange: (tab: 'annotations' | 'audit') => void;
}

const fieldClass = 'h-10 w-full rounded-lg border border-line-strong bg-white px-3 text-sm text-ink-900 focus:border-brand-500 focus:outline-none';

export const AnnotationPanel: React.FC<Props> = ({
  boxes, selectedBoxId, availableClasses, activeClassName, onSelectBox, onUpdateBox, onDeleteBox,
  onAcceptBox, onChangeActiveClass, onRegisterClass, audit, tab, onTabChange,
}) => {
  const [customClass, setCustomClass] = useState('');
  const [classError, setClassError] = useState('');
  const selected = boxes.find(box => box.id === selectedBoxId) || null;
  const showAudit = !!audit && tab === 'audit';

  const applyCustomClass = async () => {
    if (!selected || !onRegisterClass || !customClass.trim()) return;
    const box = selected;
    const name = customClass.trim();
    try {
      const id = await onRegisterClass(name);
      onUpdateBox({ ...box, class_name: availableClasses[id] || name, class_id: id });
      setCustomClass('');
      setClassError('');
    } catch (err) {
      setClassError(String(err));
    }
  };

  return (
    <aside className="flex w-[260px] shrink-0 flex-col overflow-hidden rounded-xl border border-line bg-white">
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-line px-4">
        {audit ? (
          <div className="flex gap-1" role="tablist">
            {([['annotations', 'Anotasi', boxes.length], ['audit', 'Cek kualitas', audit.count]] as const).map(([id, label, count]) => (
              <button key={id} type="button" role="tab" aria-selected={tab === id} onClick={() => onTabChange(id)}
                className={`flex items-center gap-1.5 rounded-md px-2 py-1 text-[13px] font-semibold cursor-pointer ${
                  tab === id ? 'bg-brand-50 text-ink-900' : 'text-ink-500 hover:text-ink-900'
                }`}>
                {label}
                <span className="rounded bg-brand-100 px-1.5 text-[11px] font-medium text-brand-700 tabular-nums">{count}</span>
              </button>
            ))}
          </div>
        ) : (
          <>
            <h2 className="text-[15px] font-semibold text-ink-900">Anotasi</h2>
            <span className="rounded bg-brand-100 px-1.5 py-0.5 text-[11px] font-medium text-brand-700 tabular-nums">{boxes.length}</span>
          </>
        )}
      </div>

      {showAudit ? (
        <div className="flex min-h-0 flex-1 flex-col">{audit!.content}</div>
      ) : (
        <>
          <ul className="min-h-0 flex-1 overflow-y-auto py-1" aria-label="Daftar anotasi">
            {boxes.length === 0 && (
              <li className="px-4 py-6 text-center text-xs text-ink-400">Belum ada anotasi. Gambar kotak dengan B atau jalankan Deteksi AI.</li>
            )}
            {boxes.map((box, index) => {
              const isSelected = box.id === selectedBoxId;
              const color = box.confidence < LOW_CONFIDENCE ? LOW_CONFIDENCE_COLOR : getClassColor(box.class_name);
              return (
                <li key={box.id}>
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => onSelectBox(box.id)}
                    onKeyDown={event => { if (event.key === 'Enter') onSelectBox(box.id); }}
                    className={`group flex h-10 items-center gap-2.5 px-4 text-sm cursor-pointer ${
                      isSelected ? 'bg-brand-50' : 'hover:bg-brand-50/60'
                    }`}
                  >
                    <span className="w-5 text-[11px] tabular-nums text-ink-400">{String(index + 1).padStart(2, '0')}</span>
                    <span className="h-2 w-2 shrink-0 rounded-sm" style={{ backgroundColor: color.stroke }} />
                    <span className={`min-w-0 flex-1 truncate ${isSelected ? 'font-semibold text-ink-900' : 'text-ink-700'}`}>
                      {box.class_name}
                      {box.occluded && <span className="ml-1.5 text-[10px] font-normal text-ink-400">tertutup</span>}
                    </span>
                    <span className="text-xs tabular-nums text-ink-500">{Math.round(box.confidence * 100)}%</span>
                    <button
                      type="button"
                      aria-label={`Hapus anotasi ${index + 1}`}
                      onClick={event => { event.stopPropagation(); onDeleteBox(box.id); }}
                      className="flex h-6 w-6 items-center justify-center rounded text-ink-300 hover:bg-danger-50 hover:text-danger-600 cursor-pointer"
                    >
                      <Icon name="close" className="text-[16px]" />
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>

          <div className="shrink-0 space-y-3 border-t border-line p-4">
            {selected ? (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-[13px] font-semibold text-ink-900">Objek terpilih</span>
                  <button type="button" onClick={() => onSelectBox(null)} className="text-[11px] text-ink-500 hover:text-brand-600 cursor-pointer">
                    Batal pilih
                  </button>
                </div>
                <select
                  aria-label="Kelas objek terpilih"
                  value={selected.class_name}
                  onChange={event => {
                    const index = availableClasses.indexOf(event.target.value);
                    onUpdateBox({ ...selected, class_name: event.target.value, class_id: index >= 0 ? index : 0 });
                  }}
                  className={`${fieldClass} cursor-pointer`}
                >
                  {!availableClasses.includes(selected.class_name) && <option value={selected.class_name}>{selected.class_name}</option>}
                  {availableClasses.map((name, index) => <option key={name} value={name}>[{index + 1}] {name}</option>)}
                </select>
                {onRegisterClass && (
                  <div className="flex gap-1.5">
                    <input
                      value={customClass}
                      onChange={event => setCustomClass(event.target.value)}
                      onKeyDown={event => { if (event.key === 'Enter') void applyCustomClass(); }}
                      placeholder="Kelas baru…"
                      aria-label="Nama kelas baru"
                      className="h-8 min-w-0 flex-1 rounded-md border border-line-strong px-2 text-xs focus:border-brand-500 focus:outline-none"
                    />
                    <button type="button" onClick={() => void applyCustomClass()}
                      className="h-8 rounded-md border border-line-strong px-2.5 text-xs font-semibold text-ink-700 hover:bg-brand-50 cursor-pointer">
                      Pakai
                    </button>
                  </div>
                )}
                {classError && <p role="alert" className="text-xs text-danger-600">{classError}</p>}
                <div className="flex gap-4 text-xs text-ink-700">
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input type="checkbox" checked={selected.occluded || false} className="accent-brand-600"
                      onChange={event => onUpdateBox({ ...selected, occluded: event.target.checked })} />
                    Tertutup
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input type="checkbox" checked={selected.truncated || false} className="accent-brand-600"
                      onChange={event => onUpdateBox({ ...selected, truncated: event.target.checked })} />
                    Terpotong tepi
                  </label>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <button type="button" onClick={() => onAcceptBox(selected.id)} title="Tandai kotak benar (Enter)"
                    className="flex h-9 items-center justify-center gap-1 rounded-lg border border-line-strong text-xs font-semibold text-ink-700 hover:bg-brand-50 cursor-pointer">
                    <Icon name="check" className="text-[16px]" /> Benar
                  </button>
                  <button type="button" onClick={() => onDeleteBox(selected.id)} title="Hapus kotak (Del)"
                    className="flex h-9 items-center justify-center gap-1 rounded-lg border border-line-strong text-xs font-semibold text-danger-600 hover:bg-danger-50 cursor-pointer">
                    <Icon name="delete" className="text-[16px]" /> Hapus
                  </button>
                </div>
              </>
            ) : (
              <>
                <label htmlFor="new-box-class" className="block text-[13px] font-semibold text-ink-900">Kelas kotak baru</label>
                <select
                  id="new-box-class"
                  value={activeClassName}
                  onChange={event => onChangeActiveClass(event.target.value)}
                  className={`${fieldClass} cursor-pointer`}
                >
                  {availableClasses.map((name, index) => <option key={name} value={name}>[{index + 1}] {name}</option>)}
                </select>
                <p className="text-[11px] leading-relaxed text-ink-500">
                  Pilih objek di gambar atau daftar anotasi untuk mengubah kelas.
                </p>
              </>
            )}
          </div>

          <div className="flex shrink-0 gap-2 border-t border-line px-4 py-3 text-[11px] leading-relaxed text-ink-500">
            <Icon name="warning" className="text-[15px]" />
            Confidence dari model hanya perkiraan. Periksa label sebelum menyetujui.
          </div>
        </>
      )}
    </aside>
  );
};
