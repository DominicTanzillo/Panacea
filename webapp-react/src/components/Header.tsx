import type { OverlayView } from '../App';

interface HeaderProps {
  activeOverlay: OverlayView;
  onNavigate: (view: OverlayView) => void;
  healthy: boolean;
  alertCount: number;
  cdmAlertCount: number;
  dataDate?: string;
}

export function PanaceaLogo({ size = 28 }: { size?: number }) {
  return (
    <div
      className="relative flex items-center justify-center overflow-hidden shrink-0"
      style={{ width: size, height: size, background: '#3b82f6', borderRadius: 6 }}
    >
      <span className="font-extrabold text-white" style={{ fontSize: size * 0.48 }}>P</span>
    </div>
  );
}

const NAV_ITEMS: { key: Exclude<OverlayView, null>; label: string }[] = [
  { key: 'alerts', label: 'Alerts' },
  { key: 'forecast', label: 'Forecast' },
  { key: 'dashboard', label: 'Dashboard' },
  { key: 'models', label: 'Models' },
  { key: 'about', label: 'About' },
];

export function Header({
  activeOverlay, onNavigate, healthy, alertCount, cdmAlertCount, dataDate,
}: HeaderProps) {
  return (
    <header
      className="fixed top-0 left-0 right-0 flex items-center justify-between h-12 px-6 border-b"
      style={{
        zIndex: 100,
        borderColor: '#2a2a3a',
        background: '#0c0c14',
      }}
    >
      {/* Left: Logo + name */}
      <div className="flex items-center gap-3">
        <PanaceaLogo />
        <span className="text-sm font-semibold tracking-tight text-white">PANACEA</span>
      </div>

      {/* Center: Navigation */}
      <nav className="flex items-center gap-1">
        {NAV_ITEMS.map(item => {
          const isActive = activeOverlay === item.key;
          return (
            <button
              key={item.key}
              onClick={() => onNavigate(item.key)}
              className="px-4 py-2 text-sm font-medium tracking-wide transition-colors"
              style={{
                color: isActive ? '#e8e8f0' : '#7c7c96',
                borderBottom: isActive ? '2px solid #3b82f6' : '2px solid transparent',
              }}
              onMouseEnter={e => { if (!isActive) (e.target as HTMLElement).style.color = '#b0b0c0'; }}
              onMouseLeave={e => { if (!isActive) (e.target as HTMLElement).style.color = '#7c7c96'; }}
            >
              {item.label}
              {item.key === 'alerts' && alertCount > 0 && (
                <span className="ml-1.5" style={{ color: '#ef4444' }}>
                  {cdmAlertCount > 0 ? cdmAlertCount : alertCount}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Right: Status */}
      <div className="flex items-center gap-2 text-xs" style={{ color: '#7c7c96' }}>
        <span
          className="w-2 h-2 rounded-full"
          style={{
            background: healthy ? '#22c55e' : cdmAlertCount > 0 ? '#3b82f6' : '#f59e0b',
          }}
        />
        <span>{healthy ? 'Live' : dataDate ? dataDate : 'Static'}</span>
      </div>
    </header>
  );
}
