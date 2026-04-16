import { useState } from 'react';
import type { OverlayView } from '../App';
import { GlossaryPanel } from './Glossary';

interface HeaderProps {
  activeOverlay: OverlayView;
  onNavigate: (view: OverlayView) => void;
  healthy: boolean;
  cdmAlertCount?: number;
  dataDate?: string;
}

export function PanaceaLogo({ size = 28 }: { size?: number }) {
  return (
    <div
      style={{
        width: size,
        height: size,
        background: 'linear-gradient(135deg, #3b82f6, #2563eb)',
        borderRadius: 6,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
      }}
    >
      <span style={{ fontWeight: 800, color: '#fff', fontSize: size * 0.46, lineHeight: 1 }}>P</span>
    </div>
  );
}

const NAV_ITEMS: { key: Exclude<OverlayView, null>; label: string }[] = [
  { key: 'alerts', label: 'Alerts' },
  { key: 'forecast', label: 'Forecast' },
  { key: 'models', label: 'Models' },
  { key: 'dashboard', label: 'Pipeline' },
  { key: 'about', label: 'About' },
];

export function Header({
  activeOverlay, onNavigate, healthy, cdmAlertCount, dataDate,
}: HeaderProps) {
  const [glossaryOpen, setGlossaryOpen] = useState(false);
  return (
    <header style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      zIndex: 100,
      height: 56,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 32px',
      background: 'rgba(8,8,12,0.82)',
      backdropFilter: 'blur(20px)',
      borderBottom: '1px solid rgba(255,255,255,0.06)',
    }}>
      {/* Left: Logo + wordmark */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <PanaceaLogo />
        <span style={{
          fontSize: 15,
          fontWeight: 700,
          color: '#e8e8f0',
          letterSpacing: '0.04em',
        }}>
          PANACEA
        </span>
      </div>

      {/* Center: Nav */}
      <nav style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        {NAV_ITEMS.map(item => {
          const isActive = activeOverlay === item.key;
          return (
            <button
              key={item.key}
              onClick={() => onNavigate(item.key)}
              style={{
                padding: '7px 16px',
                fontSize: 13,
                fontWeight: isActive ? 600 : 500,
                color: isActive ? '#e8e8f0' : '#7c7c96',
                background: isActive ? 'rgba(59,130,246,0.10)' : 'transparent',
                border: 'none',
                borderRadius: 8,
                cursor: 'pointer',
                transition: 'all 150ms',
                fontFamily: 'inherit',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                letterSpacing: '-0.005em',
              }}
              onMouseEnter={e => {
                if (!isActive) {
                  (e.currentTarget).style.color = '#b0b0c4';
                  (e.currentTarget).style.background = 'rgba(255,255,255,0.04)';
                }
              }}
              onMouseLeave={e => {
                if (!isActive) {
                  (e.currentTarget).style.color = '#7c7c96';
                  (e.currentTarget).style.background = 'transparent';
                }
              }}
            >
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Right: Glossary + Status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 12, color: '#7c7c96' }}>
        <button
          onClick={() => setGlossaryOpen(true)}
          aria-label="Open glossary of space terms"
          style={{
            padding: '5px 12px',
            fontSize: 12,
            fontWeight: 500,
            color: '#7c8aff',
            background: 'rgba(124,138,255,0.08)',
            border: '1px solid rgba(124,138,255,0.2)',
            borderRadius: 6,
            cursor: 'pointer',
            fontFamily: 'inherit',
          }}
        >
          Glossary
        </button>
        <div style={{
          width: 7,
          height: 7,
          borderRadius: '50%',
          background: healthy ? '#22c55e' : (cdmAlertCount ?? 0) > 0 ? '#3b82f6' : '#f59e0b',
          boxShadow: healthy ? '0 0 6px rgba(34,197,94,0.4)' : undefined,
        }} />
        <span style={{ fontWeight: 500 }}>{healthy ? 'Live' : dataDate || 'Static data'}</span>
      </div>
      {glossaryOpen && <GlossaryPanel onClose={() => setGlossaryOpen(false)} />}
    </header>
  );
}
