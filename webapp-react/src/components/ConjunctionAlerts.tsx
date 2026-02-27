import type { ScreeningPair } from '../lib/api';

interface ConjunctionAlertsProps {
  pairs: ScreeningPair[];
  visible: boolean;
  onClose: () => void;
}

const TIER_COLORS: Record<string, string> = {
  HIGH: '#ff4f5a',
  MODERATE: '#ffb84f',
  LOW: '#4fff8a',
};

function riskTier(pair: ScreeningPair): string {
  // If we have forward TCA distance, use that for tier
  if (pair.tca_min_distance_km != null) {
    if (pair.tca_min_distance_km < 1) return 'HIGH';
    if (pair.tca_min_distance_km < 10) return 'MODERATE';
  }
  if (pair.risk_score > 0.40) return 'HIGH';
  if (pair.risk_score > 0.10) return 'MODERATE';
  return 'LOW';
}

function formatTCA(hours: number): string {
  if (hours < 24) return `${hours.toFixed(1)}h`;
  const days = hours / 24;
  return `${days.toFixed(1)}d`;
}

export function ConjunctionAlerts({ pairs, visible, onClose }: ConjunctionAlertsProps) {
  if (!visible || pairs.length === 0) return null;

  const predDate = pairs[0]?.prediction_date;

  return (
    <div className="absolute right-4 top-16 w-80 max-h-[70vh] rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]/95 backdrop-blur-md shadow-2xl z-20 flex flex-col">
      <div className="flex items-center justify-between p-3 border-b border-[var(--color-border)]">
        <div>
          <h3 className="font-semibold text-sm">Conjunction Alerts</h3>
          {predDate && (
            <p className="text-[10px] text-[var(--color-text-muted)] mt-0.5">
              Screening from {predDate}
            </p>
          )}
        </div>
        <button
          onClick={onClose}
          className="w-7 h-7 flex items-center justify-center rounded-lg text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-2)] transition-colors text-base"
        >
          &times;
        </button>
      </div>

      <div className="overflow-y-auto flex-1 p-2 space-y-1.5">
        {pairs.map((pair, i) => {
          const tier = riskTier(pair);
          const color = TIER_COLORS[tier];
          return (
            <div
              key={`${pair.norad_1}-${pair.norad_2}`}
              className="rounded-lg bg-[var(--color-surface-2)] p-2.5 text-xs hover:bg-[var(--color-surface-2)]/80 transition-colors cursor-default"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-medium truncate max-w-[150px]">
                  #{i + 1} {pair.name_1}
                </span>
                <span
                  className="px-1.5 py-0.5 rounded text-[10px] font-bold"
                  style={{ background: color + '22', color }}
                >
                  {tier}
                </span>
              </div>
              <div className="text-[var(--color-text-muted)] truncate">
                vs {pair.name_2}
              </div>
              <div className="flex justify-between mt-1.5 text-[var(--color-text-muted)]">
                <span>Risk: {(pair.risk_score * 100).toFixed(1)}%</span>
                <span>{pair.altitude_km.toFixed(0)} km alt</span>
              </div>
              {pair.tca_hours != null && pair.tca_min_distance_km != null && (
                <div className="flex justify-between mt-1 text-[var(--color-text-muted)] border-t border-[var(--color-border)] pt-1">
                  <span>TCA: {formatTCA(pair.tca_hours)}</span>
                  <span>Min dist: {pair.tca_min_distance_km.toFixed(1)} km</span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="p-2 border-t border-[var(--color-border)] text-center">
        <span className="text-[10px] text-[var(--color-text-muted)]">
          Orbital proximity screening &middot; updated daily
        </span>
      </div>
    </div>
  );
}
