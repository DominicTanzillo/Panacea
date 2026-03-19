import { useEffect, useCallback } from 'react';
import type { ReactNode } from 'react';

/* ── Close button (shared) ────────────────────────────────── */
function CloseButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        width: 32, height: 32, display: 'flex', alignItems: 'center', justifyContent: 'center',
        borderRadius: 6, border: 'none', background: 'transparent', color: '#55556a',
        cursor: 'pointer', transition: 'all 150ms', flexShrink: 0,
      }}
      onMouseEnter={e => { (e.currentTarget).style.background = '#1a1a24'; (e.currentTarget).style.color = '#e8e8f0'; }}
      onMouseLeave={e => { (e.currentTarget).style.background = 'transparent'; (e.currentTarget).style.color = '#55556a'; }}
    >
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M1 1l12 12M13 1L1 13" /></svg>
    </button>
  );
}

/* ── Full-page panel (Models, Forecast, Alerts, etc.) ─────── */
interface FullPagePanelProps {
  visible: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  maxWidth?: number;
  children: ReactNode;
}

export function FullPagePanel({ visible, onClose, title, subtitle, maxWidth = 1120, children }: FullPagePanelProps) {
  const handleKey = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose();
  }, [onClose]);

  useEffect(() => {
    if (visible) {
      document.addEventListener('keydown', handleKey);
      return () => document.removeEventListener('keydown', handleKey);
    }
  }, [visible, handleKey]);

  if (!visible) return null;

  return (
    <div className="fixed inset-0 flex items-start justify-center" style={{ zIndex: 200 }}>
      <div className="absolute inset-0" style={{ background: 'rgba(0,0,0,0.70)', backdropFilter: 'blur(12px)' }} onClick={onClose} />
      <div
        className="relative w-full flex flex-col overflow-hidden"
        style={{
          maxWidth,
          maxHeight: 'calc(100vh - 48px)',
          marginTop: 24, marginBottom: 24, marginLeft: 24, marginRight: 24,
          borderRadius: 12,
          background: '#0c0c12',
          border: '1px solid #1e1e2c',
          boxShadow: '0 32px 64px rgba(0,0,0,0.6)',
          animation: 'overlayIn 200ms ease-out',
        }}
      >
        <div style={{
          padding: '24px 32px 20px', borderBottom: '1px solid #1e1e2c',
          display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexShrink: 0,
        }}>
          <div>
            <h2 style={{ fontSize: 20, fontWeight: 700, color: '#e8e8f0', letterSpacing: '-0.02em', marginBottom: 4 }}>{title}</h2>
            {subtitle && <p style={{ fontSize: 14, color: '#7c7c96' }}>{subtitle}</p>}
          </div>
          <CloseButton onClick={onClose} />
        </div>
        <div className="flex-1 overflow-y-auto" style={{ padding: '0 32px 32px' }}>{children}</div>
      </div>
    </div>
  );
}

/* ── Standard overlay (smaller content panels) ────────────── */
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

  // CRITICAL: return null when not visible to prevent z-index stacking issues
  if (!visible) return null;

  return (
    <div className="fixed inset-0 flex items-start justify-center" style={{ zIndex: 200 }}>
      {/* Backdrop */}
      <div
        className="absolute inset-0"
        style={{ background: 'rgba(0,0,0,0.60)', backdropFilter: 'blur(8px)' }}
        onClick={onClose}
      />

      {/* Content panel */}
      <div
        className="relative w-full flex flex-col overflow-hidden"
        style={{
          maxWidth,
          maxHeight: 'calc(100vh - 112px)',
          marginTop: 64,
          marginBottom: 48,
          marginLeft: 16,
          marginRight: 16,
          borderRadius: 12,
          background: '#111118',
          border: '1px solid #2a2a3a',
          boxShadow: '0 24px 48px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.02)',
          animation: 'overlayIn 200ms ease-out',
        }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between shrink-0"
          style={{ padding: '16px 24px', borderBottom: '1px solid #1e1e2c' }}
        >
          <div>
            <h2 style={{ fontSize: 16, fontWeight: 600, color: '#e8e8f0', letterSpacing: '-0.01em' }}>
              {title}
            </h2>
            {subtitle && (
              <p style={{ fontSize: 13, color: '#7c7c96', marginTop: 2 }}>{subtitle}</p>
            )}
          </div>
          <CloseButton onClick={onClose} />
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto">{children}</div>
      </div>
    </div>
  );
}
