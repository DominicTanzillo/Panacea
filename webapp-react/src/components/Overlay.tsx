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
  const handleKey = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose();
  }, [onClose]);

  useEffect(() => {
    if (visible) {
      document.addEventListener('keydown', handleKey);
      return () => document.removeEventListener('keydown', handleKey);
    }
  }, [visible, handleKey]);

  // Don't render anything when not visible — prevents z-index stacking issues
  if (!visible) return null;

  return (
    <div
      className="fixed inset-0 flex items-start justify-center"
      style={{ zIndex: 200 }}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0"
        style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
        onClick={onClose}
      />

      {/* Content panel */}
      <div
        className="relative w-full flex flex-col overflow-hidden"
        style={{
          maxWidth,
          maxHeight: 'calc(100vh - 104px)',
          marginTop: 56,
          marginBottom: 48,
          marginLeft: 16,
          marginRight: 16,
          borderRadius: 12,
          background: '#111118',
          border: '1px solid #2a2a3a',
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
          animation: 'overlayIn 250ms ease-out',
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 shrink-0" style={{ borderBottom: '1px solid #1e1e2c' }}>
          <div>
            <h2 className="text-base font-semibold tracking-tight" style={{ color: '#e8e8f0' }}>{title}</h2>
            {subtitle && (
              <p className="text-xs mt-0.5" style={{ color: '#7c7c96' }}>{subtitle}</p>
            )}
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center transition-colors"
            style={{ borderRadius: 6, color: '#7c7c96' }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = '#1a1a24'; (e.currentTarget as HTMLElement).style.color = '#e8e8f0'; }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'transparent'; (e.currentTarget as HTMLElement).style.color = '#7c7c96'; }}
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

      <style>{`
        @keyframes overlayIn {
          from { opacity: 0; transform: translateY(12px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
