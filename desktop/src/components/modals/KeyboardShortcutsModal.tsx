import React from 'react';
import { Button, Icon } from '../ui';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

interface ShortcutCategory {
  title: string;
  icon: string;
  items: { keys: string[]; description: string }[];
}

const CATEGORIES: ShortcutCategory[] = [
  {
    title: 'Alat anotasi',
    icon: 'construction',
    items: [
      { keys: ['V'], description: 'Pilih & ubah kotak' },
      { keys: ['B'], description: 'Gambar kotak' },
      { keys: ['P'], description: 'Gambar poligon' },
      { keys: ['H'], description: 'Geser kanvas' },
      { keys: ['F'], description: 'Ganti filter tampilan (tajam / kontras / hujan)' },
    ],
  },
  {
    title: 'Navigasi gambar',
    icon: 'photo_library',
    items: [
      { keys: ['A', 'atau', '←'], description: 'Gambar sebelumnya' },
      { keys: ['D', 'atau', '→'], description: 'Gambar berikutnya' },
      { keys: ['Home'], description: 'Gambar pertama' },
      { keys: ['End'], description: 'Gambar terakhir' },
    ],
  },
  {
    title: 'Kotak terpilih',
    icon: 'select',
    items: [
      { keys: ['Tab'], description: 'Pilih kotak berikutnya' },
      { keys: ['Shift', 'Tab'], description: 'Pilih kotak sebelumnya' },
      { keys: ['1-9'], description: 'Ganti kelas kotak' },
      { keys: ['Enter'], description: 'Tandai kotak benar' },
      { keys: ['Del'], description: 'Hapus kotak' },
      { keys: ['Esc'], description: 'Batal pilih / tutup dialog' },
    ],
  },
  {
    title: 'AI & cek kualitas',
    icon: 'auto_awesome',
    items: [
      { keys: ['R'], description: 'Deteksi AI pada gambar ini' },
      { keys: ['Q'], description: 'Cari objek terlewat, tertutup, dan kotak salah' },
      { keys: ['J', 'atau', 'K'], description: 'Temuan berikutnya / sebelumnya' },
      { keys: ['Y', 'atau', 'N'], description: 'Terapkan / abaikan temuan' },
    ],
  },
  {
    title: 'Halaman',
    icon: 'space_dashboard',
    items: [
      { keys: ['Ctrl', 'S'], description: 'Simpan & setujui gambar' },
      { keys: ['L'], description: 'Buka auto-label' },
      { keys: ['I'], description: 'Buka impor dataset' },
      { keys: ['E'], description: 'Buka ekspor dataset' },
      { keys: ['?'], description: 'Tampilkan pintasan ini' },
    ],
  },
];

export const KeyboardShortcutsModal: React.FC<Props> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink-900/30 p-4 backdrop-blur-[2px]" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="shortcuts-title"
        className="flex max-h-[85vh] w-[760px] flex-col overflow-hidden rounded-2xl border border-line bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-line px-6 py-4">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-100 text-brand-600">
              <Icon name="keyboard" className="text-[20px]" />
            </span>
            <div>
              <h2 id="shortcuts-title" className="text-lg font-bold text-ink-900">Pintasan keyboard</h2>
              <p className="text-xs text-ink-500">Pintasan alat dan gambar aktif di halaman review.</p>
            </div>
          </div>
          <button type="button" onClick={onClose} title="Tutup (Esc)" aria-label="Tutup"
            className="flex h-8 w-8 items-center justify-center rounded-md text-ink-500 hover:bg-brand-50 cursor-pointer">
            <Icon name="close" className="text-[18px]" />
          </button>
        </div>

        <div className="grid grid-cols-1 gap-4 overflow-y-auto p-6 md:grid-cols-2">
          {CATEGORIES.map(category => (
            <section key={category.title} className="rounded-xl border border-line p-4">
              <h3 className="mb-3 flex items-center gap-2 text-[13px] font-semibold text-ink-900">
                <Icon name={category.icon} className="text-[17px] text-brand-600" />
                {category.title}
              </h3>
              <div className="space-y-2">
                {category.items.map(item => (
                  <div key={item.description} className="flex items-center justify-between gap-3 text-xs">
                    <span className="text-ink-500">{item.description}</span>
                    <span className="flex shrink-0 items-center gap-1">
                      {item.keys.map((key, index) => key === 'atau' ? (
                        <span key={index} className="text-[10px] text-ink-400">/</span>
                      ) : (
                        <kbd key={index}
                          className="min-w-[22px] rounded border border-line-strong bg-brand-50 px-1.5 py-0.5 text-center font-sans text-[11px] font-semibold text-ink-700">
                          {key}
                        </kbd>
                      ))}
                    </span>
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>

        <div className="flex items-center justify-between border-t border-line px-6 py-3 text-xs text-ink-500">
          <span>Tekan <kbd className="rounded border border-line-strong bg-brand-50 px-1 font-sans">?</kbd> kapan saja untuk membuka panel ini.</span>
          <Button size="sm" onClick={onClose}>Mengerti</Button>
        </div>
      </div>
    </div>
  );
};
