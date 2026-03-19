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
    <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, zIndex: 20 }}>
      {/* Group toggles */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 16px', overflowX: 'auto' }}>
        {groups.map(group => (
          <button
            key={group.id}
            onClick={() => onToggleGroup(group.id)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '6px 12px',
              fontSize: 12,
              fontWeight: 500,
              fontFamily: 'inherit',
              whiteSpace: 'nowrap',
              color: group.enabled ? '#e8e8f0' : '#55556a',
              background: group.enabled ? 'rgba(59,130,246,0.08)' : 'rgba(17,17,24,0.80)',
              border: group.enabled ? '1px solid rgba(59,130,246,0.25)' : '1px solid #2a2a3a',
              borderRadius: 6,
              cursor: 'pointer',
              backdropFilter: 'blur(8px)',
              transition: 'all 150ms',
            }}
          >
            <div style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: group.enabled ? group.color : '#55556a',
            }} />
            {group.label}
          </button>
        ))}
      </div>

      {/* Status bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 16px',
        background: 'rgba(17,17,24,0.90)',
        backdropFilter: 'blur(12px)',
        borderTop: '1px solid rgba(42,42,58,0.5)',
        fontSize: 12,
        color: '#55556a',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          {loading ? (
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: '#f59e0b',
                animation: 'pulse-dot 1.5s ease-in-out infinite',
              }} />
              Loading TLEs...
            </span>
          ) : (
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e' }} />
              <span style={{ color: '#7c7c96' }}>{satellites.length.toLocaleString()} objects</span>
            </span>
          )}
          <span>{totalTLEs.toLocaleString()} TLEs</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          {byType.active && <span style={{ color: '#22c55e' }}>{byType.active.toLocaleString()} active</span>}
          {byType.debris && <span style={{ color: '#ef4444' }}>{byType.debris.toLocaleString()} debris</span>}
          {byType.station && <span style={{ color: '#e8e8f0' }}>{byType.station} stations</span>}
          {byType.constellation && <span style={{ color: '#3b82f6' }}>{byType.constellation.toLocaleString()} constellation</span>}
          {lastUpdate && (
            <span>Updated: {lastUpdate.toLocaleTimeString()}</span>
          )}
        </div>
      </div>
    </div>
  );
}
