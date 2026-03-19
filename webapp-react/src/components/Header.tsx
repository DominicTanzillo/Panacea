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
      style={{ width: size, height: size, background: 'var(--color-accent)', borderRadius: 6 }}
    >
      <span
        className="font-extrabold text-white"
        style={{ fontSize: size * 0.48 }}
      >
        P
      </span>
    </div>
  );
}

const NAV_ITEMS: { key: OverlayView; label: string }[] = [
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
    <header className="absolute top-0 left-0 right-0 z-30 pointer-events-none">
      <div
        className="relative flex items-center justify-between h-12 px-6 pointer-events-auto border-b"
        style={{
          borderColor: 'var(--color-border-subtle)',
          background: 'rgba(8,8,12,0.7)',
          backdropFilter: 'blur(12px)',
        }}
      >
        {/* Left: Logo + name */}
        <div className="flex items-center gap-3">
          <PanaceaLogo />
          <span className="text-sm font-semibold tracking-tight">PANACEA</span>
        </div>

        {/* Center: Navigation */}
        <nav className="flex items-center gap-1">
          {NAV_ITEMS.map(item => {
            const isActive = activeOverlay === item.key;
            return (
              <button
                key={item.key}
                onClick={() => onNavigate(item.key)}
                className={`px-3 py-1.5 text-xs font-medium tracking-wide transition-colors border-b-2
                  ${isActive
                    ? 'border-[var(--color-accent)] text-[var(--color-text)]'
                    : 'border-transparent text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
                  }`}
              >
                {item.label}
                {item.key === 'alerts' && alertCount > 0 && (
                  <span className="ml-1 text-[var(--color-risk-red)]">
                    {cdmAlertCount > 0 ? cdmAlertCount : alertCount}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Right: Status */}
        <div className="flex items-center gap-2 text-xs text-[var(--color-text-muted)]">
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{
              background: healthy
                ? 'var(--color-risk-green)'
                : cdmAlertCount > 0
                  ? 'var(--color-accent)'
                  : 'var(--color-risk-amber)',
            }}
          />
          <span>{healthy ? 'Live' : dataDate ? `${dataDate}` : 'Static'}</span>
        </div>
      </div>
    </header>
  );
}
