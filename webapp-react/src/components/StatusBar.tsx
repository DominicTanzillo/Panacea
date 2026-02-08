import type { ObjectGroup, SatellitePosition } from '../lib/types';

interface StatusBarProps {
  satellites: SatellitePosition[];
  loading: boolean;
  totalTLEs: number;
  groups: ObjectGroup[];
  onToggleGroup: (id: string) => void;
  lastUpdate: Date | null;
}

export function StatusBar({
  satellites,
  loading,
  totalTLEs,
  groups,
  onToggleGroup,
  lastUpdate,
}: StatusBarProps) {
  const byType = satellites.reduce((acc, s) => {
    acc[s.type] = (acc[s.type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="absolute bottom-0 left-0 right-0 z-20">
      {/* Group toggles */}
      <div className="flex items-center gap-2 px-4 py-2 overflow-x-auto">
        {groups.map(group => (
          <button
            key={group.id}
            onClick={() => onToggleGroup(group.id)}
            className={`
              flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium
              border transition-all whitespace-nowrap
              ${group.enabled
                ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-text)]'
                : 'border-[var(--color-border)] bg-[var(--color-surface)]/80 text-[var(--color-text-muted)]'
              }
            `}
          >
            <div
              className="w-2 h-2 rounded-full"
              style={{ background: group.enabled ? group.color : '#555' }}
            />
            {group.label}
          </button>
        ))}
      </div>

      {/* Status bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-[var(--color-surface)]/90 backdrop-blur-md border-t border-[var(--color-border)] text-xs text-[var(--color-text-muted)]">
        <div className="flex items-center gap-4">
          {loading ? (
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[var(--color-warning)] animate-pulse" />
              Loading TLEs...
            </span>
          ) : (
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[var(--color-safe)]" />
              {satellites.length.toLocaleString()} objects tracked
            </span>
          )}
          <span>TLEs loaded: {totalTLEs.toLocaleString()}</span>
        </div>

        <div className="flex items-center gap-4">
          {byType.active && <span className="text-[#4fff8a]">{byType.active.toLocaleString()} active</span>}
          {byType.debris && <span className="text-[#ff4f5a]">{byType.debris.toLocaleString()} debris</span>}
          {byType.station && <span className="text-white">{byType.station} stations</span>}
          {byType.constellation && <span className="text-[#4f8aff]">{byType.constellation.toLocaleString()} constellation</span>}
          {lastUpdate && (
            <span>Updated: {lastUpdate.toLocaleTimeString()}</span>
          )}
        </div>
      </div>
    </div>
  );
}
