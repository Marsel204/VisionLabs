import React from 'react';

export const APP_NAME = 'anotara';

/** Anotara wordmark: a focus reticle around a dot, then "anotara." */
export const Logo: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`flex items-center gap-2.5 select-none ${className}`} aria-label="Anotara">
    <svg viewBox="0 0 32 32" className="h-7 w-7 shrink-0" fill="none" aria-hidden="true">
      <path
        d="M4 11V7a3 3 0 0 1 3-3h4M21 4h4a3 3 0 0 1 3 3v4M28 21v4a3 3 0 0 1-3 3h-4M11 28H7a3 3 0 0 1-3-3v-4"
        stroke="#2f6fe4" strokeWidth="2.6" strokeLinecap="round"
      />
      <rect x="12" y="12" width="8" height="8" rx="2" fill="#2f6fe4" />
    </svg>
    <span className="text-[26px] leading-none font-bold tracking-tight text-ink-900">
      {APP_NAME}<span className="text-brand-600">.</span>
    </span>
  </div>
);
