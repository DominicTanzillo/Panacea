import { useState } from 'react';
import type { ScreeningPair } from '../lib/api';
import type { TLERecord, ProjectedPair } from '../lib/types';
import { AlertDetail } from './AlertDetail';

interface ConjunctionAlertsProps {
  pairs: ScreeningPair[];
  tles: TLERecord[];
  visible: boolean;
  onClose: () => void;
  onProjection: (pair: ProjectedPair | null) => void;
}

const TIER_COLORS: Record<string, string> = {
  HIGH: '#ff4f5a',
  MODERATE: '#ffb84f',
  LOW: '#4fff8a',
};

export function riskTier(pair: ScreeningPair): string {
  // CDM-based tiers: use collision probability when available
  if (pair.source === 'cdm' && pair.pc != null) {
    if (pair.pc >= 1e-4) return 'HIGH';
    if (pair.pc >= 1e-7) return 'MODERATE';
    return 'LOW';
  }
  // TLE-screening fallback: miss-distance thresholds
  if (pair.tca_min_distance_km != null) {
    if (pair.tca_min_distance_km < 1) return 'HIGH';
    if (pair.tca_min_distance_km < 5) return 'MODERATE';
    return 'LOW';
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

function formatPc(pc: number): string {
  if (pc >= 0.01) return `${(pc * 100).toFixed(1)}%`;
  if (pc >= 1e-4) return `${pc.toExponential(1)}`;
  return `${pc.toExponential(1)}`;
}

const OBJ_TYPE_SHORT: Record<string, string> = {
  PAYLOAD: 'SAT',
  ROCKET_BODY: 'R/B',
  DEBRIS: 'DEB',
  UNKNOWN: '???',
};

function shortObjType(t?: string): string {
  if (!t) return '';
  return OBJ_TYPE_SHORT[t.toUpperCase()] || t.slice(0, 3).toUpperCase();
}

export function ConjunctionAlerts({ pairs, tles, visible, onClose, onProjection }: ConjunctionAlertsProps) {
  const [selectedPair, setSelectedPair] = useState<ScreeningPair | null>(null);

  if (!visible || pairs.length === 0) return null;

  if (selectedPair) {
    return (
      <AlertDetail
        pair={selectedPair}
        tles={tles}
        onBack={() => setSelectedPair(null)}
        onProjection={onProjection}
      />
    );
  }

  const predDate = pairs[0]?.prediction_date;
  const cdmCount = pairs.filter(p => p.source === 'cdm').length;
  const tleCount = pairs.length - cdmCount;

  return (
    <div className="absolute right-4 top-16 w-80 max-h-[70vh] rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]/95 backdrop-blur-md shadow-2xl z-20 flex flex-col">
      <div className="flex items-center justify-between p-3 border-b border-[var(--color-border)]">
        <div>
          <h3 className="font-semibold text-sm">Conjunction Alerts</h3>
          <div className="flex items-center gap-2 mt-0.5">
            {predDate && (
              <span className="text-[10px] text-[var(--color-text-muted)]">
                {predDate}
              </span>
            )}
            {cdmCount > 0 && (
              <span className="text-[10px] px-1 py-0.5 rounded bg-[#8b5cf6]/15 text-[#8b5cf6] font-medium">
                {cdmCount} CDM
              </span>
            )}
            {tleCount > 0 && (
              <span className="text-[10px] px-1 py-0.5 rounded bg-[#4f8aff]/15 text-[#4f8aff] font-medium">
                {tleCount} TLE
              </span>
            )}
          </div>
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
          const isCDM = pair.source === 'cdm';
          return (
            <div
              key={`${pair.norad_1}-${pair.norad_2}`}
              onClick={() => setSelectedPair(pair)}
              className="rounded-lg bg-[var(--color-surface-2)] p-2.5 text-xs hover:bg-[var(--color-surface-2)]/60 transition-colors cursor-pointer group"
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="font-medium truncate max-w-[120px]">
                    #{i + 1} {pair.name_1}
                  </span>
                  {isCDM && pair.emergency_reportable === 'Y' && (
                    <span className="px-1 py-0.5 rounded text-[9px] font-bold bg-[#ff4f5a]/15 text-[#ff4f5a] shrink-0">
                      EMRG
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  {isCDM && (
                    <span className="px-1 py-0.5 rounded text-[9px] font-semibold bg-[#8b5cf6]/15 text-[#8b5cf6]">
                      CDM
                    </span>
                  )}
                  <span
                    className="px-1.5 py-0.5 rounded text-[10px] font-bold"
                    style={{ background: color + '22', color }}
                  >
                    {tier}
                  </span>
                  <span className="text-[var(--color-text-muted)] opacity-0 group-hover:opacity-100 transition-opacity text-[10px]">
                    &rarr;
                  </span>
                </div>
              </div>
              <div className="text-[var(--color-text-muted)] truncate">
                vs {pair.name_2}
              </div>

              {/* CDM-specific metrics row */}
              {isCDM && pair.pc != null && (
                <div className="flex items-center gap-3 mt-1.5 text-[var(--color-text-muted)]">
                  <span>
                    Pc: <span className="font-mono font-semibold" style={{ color }}>
                      {formatPc(pair.pc)}
                    </span>
                  </span>
                  {pair.miss_distance_km != null && (
                    <span>Miss: {pair.miss_distance_km.toFixed(1)} km</span>
                  )}
                  {pair.sat1_type && pair.sat2_type && (
                    <span className="ml-auto text-[10px]">
                      {shortObjType(pair.sat1_type)}  vs {shortObjType(pair.sat2_type)}
                    </span>
                  )}
                </div>
              )}

              {/* TLE-screening metrics row */}
              {!isCDM && (
                <div className="flex justify-between mt-1.5 text-[var(--color-text-muted)]">
                  <span>{pair.altitude_km.toFixed(0)} km alt</span>
                  {pair.tca_min_distance_km != null && (
                    <span>Min dist: {pair.tca_min_distance_km.toFixed(1)} km</span>
                  )}
                </div>
              )}

              {pair.tca_hours != null && (
                <div className="flex justify-between mt-1 text-[var(--color-text-muted)] border-t border-[var(--color-border)] pt-1">
                  <span>TCA: {formatTCA(pair.tca_hours)}</span>
                  {isCDM && pair.cdm_tca && (
                    <span className="text-[10px]">
                      {new Date(pair.cdm_tca).toISOString().slice(5, 16).replace('T', ' ')}
                    </span>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="p-2 border-t border-[var(--color-border)] text-center">
        <span className="text-[10px] text-[var(--color-text-muted)]">
          {cdmCount > 0 ? 'Space-Track CDM data' : 'Orbital proximity screening'} &middot; click pair for trajectory
        </span>
      </div>
    </div>
  );
}
