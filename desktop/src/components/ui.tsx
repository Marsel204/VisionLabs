import React from 'react';

/** Shared Anotara building blocks: keep pages visually consistent with the design reference. */

export const Icon: React.FC<{ name: string; className?: string; title?: string }> = ({ name, className = '', title }) => (
  <span className={`material-symbols-outlined ${className}`} aria-hidden={title ? undefined : true} title={title}>
    {name}
  </span>
);

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'soft' | 'danger';

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: 'bg-brand-600 text-white hover:bg-brand-700 border border-brand-600 hover:border-brand-700 shadow-sm',
  secondary: 'bg-white text-ink-900 border border-line-strong hover:bg-brand-50 hover:border-brand-300',
  ghost: 'text-ink-500 hover:text-brand-600 hover:bg-brand-50 border border-transparent',
  soft: 'bg-brand-50 text-brand-600 border border-brand-200 hover:bg-brand-100',
  danger: 'bg-white text-danger-600 border border-line-strong hover:bg-danger-50 hover:border-danger-600/40',
};

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: 'sm' | 'md' | 'lg';
  icon?: string;
  trailingIcon?: string;
  block?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary', size = 'md', icon, trailingIcon, block, className = '', children, type = 'button', ...rest
}) => {
  const sizes = { sm: 'h-8 px-3 text-xs gap-1.5', md: 'h-10 px-4 text-sm gap-2', lg: 'h-11 px-5 text-sm gap-2' };
  return (
    <button
      type={type}
      className={`inline-flex items-center justify-center rounded-lg font-semibold transition-colors cursor-pointer
        disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2
        focus-visible:outline-brand-600 ${sizes[size]} ${BUTTON_VARIANTS[variant]} ${block ? 'w-full' : ''} ${className}`}
      {...rest}
    >
      {icon && <Icon name={icon} className="text-[18px]" />}
      {children}
      {trailingIcon && <Icon name={trailingIcon} className="text-[18px]" />}
    </button>
  );
};

export const Card: React.FC<React.HTMLAttributes<HTMLDivElement> & { tone?: 'white' | 'soft' }> = ({
  tone = 'white', className = '', children, ...rest
}) => (
  <div
    className={`rounded-xl border ${tone === 'soft' ? 'bg-brand-50 border-brand-200' : 'bg-white border-line'} ${className}`}
    {...rest}
  >
    {children}
  </div>
);

type BadgeTone = 'warn' | 'brand' | 'neutral' | 'ok' | 'danger';
const BADGE_TONES: Record<BadgeTone, string> = {
  warn: 'bg-warn-50 text-warn-700',
  brand: 'bg-brand-100 text-brand-700',
  neutral: 'bg-[#eef1f6] text-ink-700',
  ok: 'bg-ok-50 text-ok-600',
  danger: 'bg-danger-50 text-danger-600',
};

export const Badge: React.FC<{ tone?: BadgeTone; className?: string; children: React.ReactNode }> = ({
  tone = 'neutral', className = '', children,
}) => (
  <span className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium ${BADGE_TONES[tone]} ${className}`}>
    {children}
  </span>
);

export const PageHeader: React.FC<{ title: React.ReactNode; subtitle?: React.ReactNode; actions?: React.ReactNode }> = ({
  title, subtitle, actions,
}) => (
  <div className="flex flex-wrap items-start justify-between gap-4 mb-8">
    <div className="min-w-0">
      <h1 className="text-[28px] leading-tight font-bold tracking-tight text-ink-900">{title}</h1>
      {subtitle && <p className="mt-2 text-sm text-ink-500">{subtitle}</p>}
    </div>
    {actions && <div className="flex items-center gap-3">{actions}</div>}
  </div>
);

export const ProgressBar: React.FC<{ value: number; className?: string; tone?: 'brand' | 'warn' }> = ({
  value, className = '', tone = 'brand',
}) => (
  <div className={`h-1.5 w-full rounded-full bg-brand-100 overflow-hidden ${className}`}>
    <div
      className={`h-full rounded-full transition-all duration-300 ${tone === 'warn' ? 'bg-warn-500' : 'bg-brand-600'}`}
      style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
    />
  </div>
);

/** A calm information strip with a shield icon, used for privacy and safety notes. */
export const Note: React.FC<{ title?: React.ReactNode; children?: React.ReactNode; icon?: string; className?: string }> = ({
  title, children, icon = 'verified_user', className = '',
}) => (
  <div className={`flex gap-3 rounded-lg bg-brand-50 px-4 py-3 ${className}`}>
    <Icon name={icon} className="text-[18px] text-ink-500 mt-0.5" />
    <div className="min-w-0 text-xs leading-relaxed">
      {title && <div className="font-semibold text-ink-900 text-[13px]">{title}</div>}
      {children && <div className="text-ink-500 mt-0.5">{children}</div>}
    </div>
  </div>
);

export const KeyHint: React.FC<{ k: string; label?: string }> = ({ k, label }) => (
  <span className="inline-flex items-center gap-1.5 text-[11px] text-ink-500">
    <kbd className="inline-flex h-5 min-w-5 items-center justify-center rounded border border-line-strong bg-white px-1 font-sans text-[10px] font-semibold text-ink-700">
      {k}
    </kbd>
    {label}
  </span>
);

export const StatRow: React.FC<{ label: React.ReactNode; value: React.ReactNode; dot?: string }> = ({ label, value, dot }) => (
  <div className="flex items-center justify-between py-1.5 text-sm">
    <span className="flex items-center gap-2 text-ink-500">
      {dot && <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: dot }} />}
      {label}
    </span>
    <span className="font-semibold text-ink-900 tabular-nums">{value}</span>
  </div>
);
