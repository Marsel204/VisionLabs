import React, { useState } from 'react';
import type { ClassChange, ClassChangeResult } from '../services/api';
import { Button } from '../components/ui';

interface Props {
  classes: string[];
  datasetId: string;
  onManage: (change: ClassChange) => Promise<ClassChangeResult>;
}

export const ClassManager: React.FC<Props> = ({ classes, datasetId, onManage }) => {
  const [editing, setEditing] = useState<{ name: string; action: 'rename' | 'delete' } | null>(null);
  const [name, setName] = useState('');
  const [strategy, setStrategy] = useState('unused');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const start = (className: string, action: 'rename' | 'delete') => {
    setEditing({ name: className, action });
    setName(className);
    setStrategy('unused');
    setError(null);
    setMessage(null);
  };

  const apply = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!editing || busy) return;
    setBusy(true);
    setError(null);
    try {
      const result = await onManage({
        dataset_id: datasetId, expected_classes: classes,
        action: editing.action, name: editing.name,
        ...(editing.action === 'rename' ? { target: name.trim() } : {
          ...(strategy.startsWith('move:') ? { target: strategy.slice(5) } : {}),
          delete_annotations: strategy === 'delete',
        }),
      });
      setEditing(null);
      setMessage(`Kelas berhasil ${editing.action === 'rename' ? 'diubah' : 'dihapus'}. Backup tersimpan di ${result.backup}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const duplicate = classes.some(item => item !== editing?.name && item.toLowerCase() === name.trim().toLowerCase());

  return (
    <div className="mt-2">
      {classes.length === 0 && <p className="text-xs text-ink-500">Belum ada kelas.</p>}
      <ul className="divide-y divide-line">
        {classes.map((className, index) => (
          <li key={className} className="py-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="min-w-0 flex-1 break-words text-sm font-medium text-ink-900">{className}</span>
              <span className="text-xs text-ink-500">ID {index}</span>
              <Button size="sm" variant="ghost" disabled={busy || !datasetId}
                aria-label={`Ubah nama kelas ${className}`} onClick={() => start(className, 'rename')}>Ubah</Button>
              <Button size="sm" variant="danger" disabled={busy || !datasetId}
                aria-label={`Hapus kelas ${className}`} onClick={() => start(className, 'delete')}>Hapus</Button>
            </div>
            {editing?.name === className && (
              <form onSubmit={event => void apply(event)} className="mt-3 space-y-3 rounded-lg border border-line-strong bg-brand-50 p-3"
                aria-label={`${editing.action === 'rename' ? 'Ubah nama' : 'Hapus'} kelas ${className}`}>
                {editing.action === 'rename' ? (
                  <label className="block text-xs font-medium text-ink-700">
                    Nama baru
                    <input autoFocus value={name} disabled={busy} onChange={event => setName(event.target.value)}
                      className="mt-1 h-9 w-full rounded-lg border border-line-strong bg-white px-3 text-sm focus:border-brand-500 focus:outline-none"
                      aria-invalid={duplicate} />
                    {duplicate && <span className="mt-1 block text-danger-600">Nama ini sudah digunakan.</span>}
                  </label>
                ) : (
                  <label className="block text-xs font-medium text-ink-700">
                    Anotasi yang memakai kelas ini
                    <select value={strategy} disabled={busy} onChange={event => setStrategy(event.target.value)}
                      className="mt-1 min-h-9 w-full rounded-lg border border-line-strong bg-white px-2 py-2 text-xs focus:border-brand-500 focus:outline-none">
                      <option value="unused">Hapus hanya jika kelas belum digunakan</option>
                      {classes.filter(item => item !== className).map(item => <option key={item} value={`move:${item}`}>Pindahkan ke {item}</option>)}
                      <option value="delete">Hapus semua anotasi kelas ini</option>
                    </select>
                  </label>
                )}
                <p className={`text-xs leading-relaxed ${strategy === 'delete' && editing.action === 'delete' ? 'text-danger-600' : 'text-ink-700'}`}>
                  {editing.action === 'rename' ? 'Nama pada anotasi tersimpan ikut diperbarui. ' : strategy === 'delete'
                    ? 'Semua box dan poligon kelas ini akan dihapus. Gambar tetap tersimpan. '
                    : 'ID kelas lain disesuaikan otomatis. Gambar tetap tersimpan. '}
                  Backup dibuat sebelum perubahan.
                </p>
                {error && <p role="alert" className="break-words text-xs text-danger-600">{error}</p>}
                <div className="flex flex-wrap gap-2">
                  <Button type="button" size="sm" variant="secondary" disabled={busy} onClick={() => { setEditing(null); setError(null); }}>Batal</Button>
                  <Button type="submit" size="sm" variant={editing.action === 'delete' ? 'danger' : 'primary'}
                    disabled={busy || (editing.action === 'rename' && (!name.trim() || duplicate))}>
                    {busy ? 'Menyimpan…' : editing.action === 'rename' ? 'Simpan nama' : 'Konfirmasi hapus'}
                  </Button>
                </div>
              </form>
            )}
          </li>
        ))}
      </ul>
      {message && <p role="status" className="mt-3 break-all text-xs leading-relaxed text-ink-700">{message}</p>}
    </div>
  );
};
