import { useEffect, useCallback } from 'react';
import type { ReactNode } from 'react';

interface OverlayProps {
  visible: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  maxWidth?: string;
  children: ReactNode;
}

export function Overlay({ visible, onClose, title, subtitle, maxWidth = '640px', children }: OverlayProps) {
  // ESC to close
  const handleKey = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose();
  }, [onClose]);

  useEffect(() => {
    if (visible) {
      document.addEventListener('keydown', handleKey);
      return () => document.removeEventListener('keydown', handleKey);
    }
  }, [visible, handleKey]);

  return (
    <div
      className={`fixed inset-0 z-40 flex items-start justify-center
        transition-all duration-300
        ${visible ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}
      style={{ transitionTimingFunction: 'cubic-bezier(0.32, 0.72, 0, 1)' }}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Content panel */}
      <div
        className={`relative mt-14 mb-12 w-full flex flex-col
          bg-[var(--color-surface)] border border-[var(--color-border)]
          shadow-2xl overflow-hidden
          transition-transform duration-300
          ${visible ? 'translate-y-0' : 'translate-y-4'}`}
        style={{
          maxWidth,
          maxHeight: 'calc(100vh - 104px)',
          borderRadius: 12,
          transitionTimingFunction: 'cubic-bezier(0.32, 0.72, 0, 1)',
        }}
      >
        {/* Sticky header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--color-border-subtle)] shrink-0">
          <div>
            <h2 className="text-base font-semibold tracking-tight">{title}</h2>
            {subtitle && (
              <p className="text-xs text-[var(--color-text-muted)] mt-0.5">{subtitle}</p>
            )}
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-2)] transition-colors"
            style={{ borderRadius: 6 }}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M1 1l12 12M13 1L1 13" />
            </svg>
          </button>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto">
          {children}
        </div>
      </div>
    </div>
  );
}
