interface HeaderProps {
  healthy: boolean;
  showBorders: boolean;
  onToggleBorders: () => void;
  showAlerts: boolean;
  onToggleAlerts: () => void;
  showDashboard: boolean;
  onToggleDashboard: () => void;
  onShowAbout: () => void;
  alertCount: number;
}

export function PanaceaLogo({ size = 32 }: { size?: number }) {
  return (
    <div
      className="relative flex items-center justify-center rounded-lg overflow-hidden shrink-0"
      style={{ width: size, height: size, background: 'linear-gradient(135deg, #4f8aff, #8b5cf6)' }}
    >
      <span
        className="font-extrabold text-white relative z-10"
        style={{ fontSize: size * 0.44 }}
      >
        P
      </span>
      <svg
        className="absolute inset-0"
        width={size}
        height={size}
        viewBox="0 0 32 32"
        fill="none"
      >
        <path
          d="M5 24 Q16 4, 28 14"
          stroke="white"
          strokeWidth="0.8"
          opacity="0.35"
          fill="none"
          strokeLinecap="round"
        />
        <circle cx="27" cy="13" r="1.5" fill="white" opacity="0.55" />
      </svg>
    </div>
  );
}

const pill =
  'px-3 py-1.5 rounded-full text-xs font-medium border transition-all backdrop-blur-md';
const pillOn =
  'border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-text)]';
const pillOff =
  'border-[var(--color-border)] bg-[var(--color-surface)]/90 text-[var(--color-text-muted)] hover:text-[var(--color-text)]';

export function Header({
  healthy,
  showBorders,
  onToggleBorders,
  showAlerts,
  onToggleAlerts,
  showDashboard,
  onToggleDashboard,
  onShowAbout,
  alertCount,
}: HeaderProps) {
  return (
    <header className="absolute top-0 left-0 right-0 z-30 pointer-events-none">
      {/* Gradient fade behind header */}
      <div className="absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-[var(--color-bg)] to-transparent" />

      <div className="relative flex items-start justify-between px-4 py-3 gap-6">
        {/* Left: Branding */}
        <div className="flex items-center gap-3 pointer-events-auto shrink-0">
          <PanaceaLogo />
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold tracking-tight">PANACEA</h1>
              <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold bg-[var(--color-surface-2)]/80 border border-[var(--color-border)] text-[var(--color-text-muted)] whitespace-nowrap">
                AIPI 540
              </span>
            </div>
            <p className="text-[10px] text-[var(--color-text-muted)] -mt-0.5 tracking-wide hidden sm:block">
              Predictive Assessment Network for Automated Conjunction Evaluation &amp; Avoidance
            </p>
          </div>
        </div>

        {/* Right: Controls */}
        <div className="flex items-center gap-2 pointer-events-auto flex-wrap justify-end">
          <div
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-[11px] border border-[var(--color-border)] bg-[var(--color-surface)]/90 backdrop-blur-md"
            title={healthy ? 'Backend connected' : 'Backend offline — globe still works'}
          >
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: healthy ? '#4fff8a' : '#ff4f5a' }}
            />
            {healthy ? 'API' : 'Offline'}
          </div>

          <button
            onClick={onToggleBorders}
            className={`${pill} ${showBorders ? pillOn : pillOff}`}
            title="Toggle country borders"
          >
            Borders
          </button>

          <button
            onClick={onToggleAlerts}
            className={`${pill} ${showAlerts ? pillOn : pillOff}`}
          >
            Alerts{alertCount > 0 ? ` (${alertCount})` : ''}
          </button>

          <button
            onClick={onToggleDashboard}
            className={`${pill} ${showDashboard ? pillOn : pillOff}`}
          >
            Dashboard
          </button>

          <button
            onClick={onShowAbout}
            className={`${pill} ${pillOff}`}
          >
            About
          </button>
        </div>
      </div>
    </header>
  );
}
