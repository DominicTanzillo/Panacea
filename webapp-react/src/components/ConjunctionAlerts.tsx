import type { ScreeningPair } from '../lib/api';
import { FullPagePanel } from './Overlay';

interface ConjunctionAlertsProps {
  pairs: ScreeningPair[];
  visible: boolean;
  onClose: () => void;
  onSelectPair: (pair: ScreeningPair) => void;
}

const TIER_COLORS: Record<string, string> = {
  HIGH: '#ef4444',
  MODERATE: '#f59e0b',
  LOW: '#22c55e',
};

export function riskTier(pair: ScreeningPair): string {
  if (pair.source === 'cdm' && pair.pc != null) {
    if (pair.pc >= 1e-4) return 'HIGH';
    if (pair.pc >= 1e-7) return 'MODERATE';
    return 'LOW';
  }
  if (pair.tca_min_distance_km != null) {
    if (pair.tca_min_distance_km < 1) return 'HIGH';
    if (pair.tca_min_distance_km < 5) return 'MODERATE';
    return 'LOW';
  }
  if (pair.risk_score > 0.40) return 'HIGH';
  if (pair.risk_score > 0.10) return 'MODERATE';
  return 'LOW';
}

function formatPc(pc: number): string {
  if (!pc || pc <= 0) return '--';
  if (pc >= 0.01) return `${(pc * 100).toFixed(1)}%`;
  if (pc >= 1e-4) {
    const ratio = Math.round(1 / pc);
    return `1 in ${ratio.toLocaleString()}`;
  }
  return pc.toExponential(1);
}

function formatTCA(hours: number): string {
  if (hours < 24) return `${hours.toFixed(1)}h`;
  return `${(hours / 24).toFixed(1)}d`;
}

const OBJ_TYPE_SHORT: Record<string, string> = { PAYLOAD: 'SAT', ROCKET_BODY: 'R/B', DEBRIS: 'DEB', UNKNOWN: '???' };
function shortObjType(t?: string): string {
  if (!t) return '';
  return OBJ_TYPE_SHORT[t.toUpperCase()] || t.slice(0, 3).toUpperCase();
}

export function ConjunctionAlerts({ pairs, visible, onClose, onSelectPair }: ConjunctionAlertsProps) {
  const cdmCount = pairs.filter(p => p.source === 'cdm').length;
  const highCount = pairs.filter(p => riskTier(p) === 'HIGH').length;
  const modCount = pairs.filter(p => riskTier(p) === 'MODERATE').length;

  const sorted = [...pairs].sort((a, b) => {
    const order: Record<string, number> = { HIGH: 0, MODERATE: 1, LOW: 2 };
    return (order[riskTier(a)] ?? 2) - (order[riskTier(b)] ?? 2);
  });

  return (
    <FullPagePanel
      visible={visible}
      onClose={onClose}
      title="Active Conjunctions"
      subtitle={`${pairs.length} pairs monitored${cdmCount > 0 ? ` \u00b7 ${cdmCount} from Space-Track CDMs` : ''}`}
      maxWidth={900}
    >
      {pairs.length === 0 ? (
        <div style={{ padding: 48, textAlign: 'center', fontSize: 14, color: '#7c7c96' }}>
          No active conjunction alerts. The pipeline screens satellite pairs daily — alerts appear when Pc exceeds screening thresholds.
        </div>
      ) : (
        <>
          {/* Summary strip */}
          <div style={{ display: 'flex', gap: 24, padding: '20px 0', borderBottom: '1px solid #1e1e2c', fontSize: 14 }}>
            {highCount > 0 && <span style={{ color: '#ef4444', fontWeight: 600 }}>{highCount} High Risk</span>}
            {modCount > 0 && <span style={{ color: '#f59e0b', fontWeight: 600 }}>{modCount} Moderate</span>}
            <span style={{ color: '#55556a' }}>{pairs.length - highCount - modCount} Low</span>
          </div>

          {/* Table header */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '4px 1fr 100px 100px 80px 72px 32px',
            gap: 12,
            padding: '12px 16px',
            fontSize: 12,
            fontWeight: 600,
            color: '#55556a',
            textTransform: 'uppercase',
            letterSpacing: '0.04em',
            borderBottom: '1px solid #1e1e2c',
          }}>
            <span />
            <span>Conjunction Pair</span>
            <span style={{ textAlign: 'right' }}>Pc</span>
            <span style={{ textAlign: 'right' }}>Miss Dist</span>
            <span style={{ textAlign: 'right' }}>TCA</span>
            <span style={{ textAlign: 'center' }}>Risk</span>
            <span />
          </div>

          {/* Pair rows */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {sorted.map((pair) => {
              const tier = riskTier(pair);
              const color = TIER_COLORS[tier];
              const isCDM = pair.source === 'cdm';

              return (
                <button
                  key={`${pair.norad_1}-${pair.norad_2}`}
                  onClick={() => onSelectPair(pair)}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '4px 1fr 100px 100px 80px 72px 32px',
                    gap: 12,
                    alignItems: 'center',
                    padding: '12px 16px',
                    background: 'transparent',
                    border: 'none',
                    borderBottom: '1px solid #111118',
                    cursor: 'pointer',
                    textAlign: 'left',
                    fontFamily: 'inherit',
                    transition: 'background 100ms',
                    width: '100%',
                  }}
                  onMouseEnter={e => { (e.currentTarget).style.background = '#111118'; }}
                  onMouseLeave={e => { (e.currentTarget).style.background = 'transparent'; }}
                >
                  {/* Risk bar */}
                  <div style={{ width: 4, height: 32, borderRadius: 2, background: color }} />

                  {/* Names */}
                  <div style={{ minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontSize: 14, fontWeight: 500, color: '#e8e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {pair.name_1}
                      </span>
                      {isCDM && (
                        <span style={{ fontSize: 12, fontWeight: 600, color: '#3b82f6', background: 'rgba(59,130,246,0.08)', padding: '1px 6px', borderRadius: 4, flexShrink: 0 }}>CDM</span>
                      )}
                      {isCDM && pair.emergency_reportable === 'Y' && (
                        <span style={{ fontSize: 12, fontWeight: 700, color: '#ef4444', background: 'rgba(239,68,68,0.08)', padding: '1px 6px', borderRadius: 4, flexShrink: 0 }}>EMRG</span>
                      )}
                    </div>
                    <div style={{ fontSize: 13, color: '#55556a', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      vs {pair.name_2}
                      {isCDM && pair.sat1_type && pair.sat2_type && (
                        <span style={{ marginLeft: 8, color: '#3a3a4a' }}>{shortObjType(pair.sat1_type)} / {shortObjType(pair.sat2_type)}</span>
                      )}
                    </div>
                  </div>

                  {/* Pc */}
                  <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: 14, fontWeight: 600, color: isCDM && pair.pc ? color : '#55556a' }}>
                    {isCDM && pair.pc != null ? formatPc(pair.pc) : '--'}
                  </div>

                  {/* Miss distance */}
                  <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: 14, color: '#7c7c96' }}>
                    {pair.miss_distance_km != null ? `${pair.miss_distance_km.toFixed(1)} km` : '--'}
                  </div>

                  {/* TCA */}
                  <div style={{ textAlign: 'right', fontSize: 13, color: '#7c7c96' }}>
                    {pair.tca_hours != null ? formatTCA(pair.tca_hours) : '--'}
                  </div>

                  {/* Risk badge */}
                  <div style={{ textAlign: 'center' }}>
                    <span style={{ fontSize: 12, fontWeight: 700, color, background: `${color}12`, padding: '3px 8px', borderRadius: 6 }}>
                      {tier}
                    </span>
                  </div>

                  {/* Arrow */}
                  <span style={{ color: '#3a3a4a', fontSize: 14, textAlign: 'center' }}>&rarr;</span>
                </button>
              );
            })}
          </div>

          {/* Footer */}
          <div style={{ padding: '16px 0', textAlign: 'center', fontSize: 13, color: '#55556a' }}>
            {cdmCount > 0 ? 'Space-Track CDM data' : 'Orbital proximity screening'} &middot; click a pair to view trajectory on globe
          </div>
        </>
      )}
    </FullPagePanel>
  );
}
