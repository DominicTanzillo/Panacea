import { useState, useEffect, useMemo, useCallback } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, ReferenceLine, ResponsiveContainer,
} from 'recharts';
import type { ScreeningPair } from '../lib/api';
import { FullPagePanel } from './Overlay';

interface ConjunctionAlertsProps {
  pairs: ScreeningPair[];
  visible: boolean;
  onClose: () => void;
  onSelectPair: (pair: ScreeningPair) => void;
}

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

function formatLeadTime(hours: number): string {
  if (hours >= 48) return `${(hours / 24).toFixed(1)} days`;
  if (hours >= 1) return `${hours.toFixed(0)}h`;
  return `${(hours * 60).toFixed(0)} min`;
}

/* eslint-disable @typescript-eslint/no-explicit-any */

interface FeaturedCall {
  sat1_name: string;
  sat2_name: string;
  sat1_norad: number;
  sat2_norad: number;
  tca: string;
  current_pc: number;
  forecast_pc: number;
  ensemble_probability: number;
  risk_direction: string;
  n_updates: number;
  prediction_lead_hours: number;
  current_miss_km: number;
  call_type: 'escalation' | 'safe';
  time_series: { update_idx: number; log10_pc: number; pc: number; miss_distance_km: number; time_to_tca_hours: number }[];
}

/** Convert a FeaturedCall to a ScreeningPair so AlertDetail + Globe can display it.
 *  If alertPairs are available, try to match by NORAD IDs to inject trajectory data. */
function featuredToScreeningPair(call: FeaturedCall, alertPairs?: ScreeningPair[]): ScreeningPair {
  const base: ScreeningPair = {
    name_1: call.sat1_name,
    name_2: call.sat2_name,
    norad_1: call.sat1_norad,
    norad_2: call.sat2_norad,
    risk_score: call.current_pc,
    altitude_km: 0,
    miss_estimate_km: call.current_miss_km,
    source: 'cdm',
    pc: call.current_pc,
    miss_distance_km: call.current_miss_km,
    cdm_tca: call.tca,
  };
  // Use trajectory data from featured call itself (future pipeline runs)
  const callAny = call as any;
  if (callAny.trajectory) {
    base.trajectory = callAny.trajectory;
    base.trail = callAny.trail;
  }
  // Fallback: match against alert pairs by NORAD IDs
  if (!base.trajectory && alertPairs) {
    const match = alertPairs.find(p =>
      (p.norad_1 === call.sat1_norad && p.norad_2 === call.sat2_norad) ||
      (p.norad_1 === call.sat2_norad && p.norad_2 === call.sat1_norad)
    );
    if (match) {
      base.trajectory = match.trajectory;
      base.trail = match.trail;
      base.tle1 = match.tle1;
      base.tle2 = match.tle2;
    }
  }
  return base;
}

function FeaturedCallCard({ call, onClick }: { call: FeaturedCall; onClick: () => void }) {
  const isEscalation = call.call_type === 'escalation';
  const color = isEscalation ? '#ef4444' : '#22c55e';

  // Build sparkline data from time_series
  const sparkData = useMemo(() => {
    if (!call.time_series || call.time_series.length < 2) return [];
    return call.time_series
      .slice()
      .sort((a, b) => b.time_to_tca_hours - a.time_to_tca_hours)
      .map(u => ({ h: -Math.round(u.time_to_tca_hours), pc: u.log10_pc }));
  }, [call.time_series]);

  return (
    <button
      onClick={onClick}
      style={{
        background: '#111118', borderRadius: 10, border: '1px solid #1e1e2c',
        overflow: 'hidden', cursor: 'pointer', textAlign: 'left',
        fontFamily: 'inherit', width: '100%', padding: 0,
        transition: 'border-color 150ms, background 150ms',
      }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = '#2a2a3a'; e.currentTarget.style.background = '#131320'; }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = '#1e1e2c'; e.currentTarget.style.background = '#111118'; }}
    >
      {/* Verdict banner */}
      <div style={{ padding: '6px 16px', background: `${color}0a`, borderBottom: `1px solid ${color}18`, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ fontSize: 14 }}>{isEscalation ? '\u2713' : '\u2713'}</span>
        <span style={{ fontSize: 12, fontWeight: 600, color }}>
          Predicted {isEscalation ? 'escalation' : 'safe'} {formatLeadTime(call.prediction_lead_hours)} before TCA — Confirmed
        </span>
      </div>

      <div style={{ padding: '12px 16px 14px' }}>
        {/* Names */}
        <div style={{ fontSize: 14, fontWeight: 600, color: '#e8e8f0', marginBottom: 8, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {call.sat1_name} <span style={{ color: '#3a3a4a', fontWeight: 400 }}>vs</span> {call.sat2_name}
        </div>

        {/* Pc sparkline */}
        {sparkData.length >= 2 && (
          <div style={{ marginBottom: 10 }}>
            <ResponsiveContainer width="100%" height={64}>
              <AreaChart data={sparkData} margin={{ top: 2, right: 4, left: 4, bottom: 2 }}>
                <defs>
                  <linearGradient id={`spark-${call.sat1_norad}-${call.sat2_norad}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#e8e8f0" stopOpacity={0.15} />
                    <stop offset="100%" stopColor="#e8e8f0" stopOpacity={0.01} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="h" hide />
                <YAxis hide domain={[-5, -1.5]} />
                <ReferenceLine y={-3.3} stroke="#ef4444" strokeDasharray="3 3" strokeWidth={1} strokeOpacity={0.5} />
                <Area type="monotone" dataKey="pc" stroke="#e8e8f0" strokeWidth={1.5} fill={`url(#spark-${call.sat1_norad}-${call.sat2_norad})`} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#3a3a4a', marginTop: 1, padding: '0 4px' }}>
              <span>{sparkData[0]?.h}h</span>
              <span style={{ color: '#55556a' }}>Pc over time</span>
              <span>TCA</span>
            </div>
          </div>
        )}

        {/* Stats grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, fontSize: 12 }}>
          <div>
            <div style={{ color: '#3a3a4a', fontSize: 10, marginBottom: 2 }}>Final Pc</div>
            <div style={{ color: '#e8e8f0', fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>{formatPc(call.current_pc)}</div>
          </div>
          <div>
            <div style={{ color: '#3a3a4a', fontSize: 10, marginBottom: 2 }}>Miss</div>
            <div style={{ color: '#e8e8f0', fontFamily: "'JetBrains Mono', monospace" }}>{call.current_miss_km.toFixed(0)} km</div>
          </div>
          <div>
            <div style={{ color: '#3a3a4a', fontSize: 10, marginBottom: 2 }}>Confidence</div>
            <div style={{ color: '#e8e8f0', fontFamily: "'JetBrains Mono', monospace" }}>{(call.ensemble_probability * 100).toFixed(0)}%</div>
          </div>
          <div>
            <div style={{ color: '#3a3a4a', fontSize: 10, marginBottom: 2 }}>CDMs</div>
            <div style={{ color: '#e8e8f0', fontFamily: "'JetBrains Mono', monospace" }}>{call.n_updates}</div>
          </div>
        </div>

        {/* TCA + click hint */}
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10, fontSize: 11, color: '#3a3a4a' }}>
          <span>TCA: {call.tca.slice(0, 16).replace('T', ' ')} UTC</span>
          <span style={{ color: '#3b82f6' }}>{(call as any).trajectory ? 'View fly-by' : 'View detail'} &rarr;</span>
        </div>
      </div>
    </button>
  );
}

export function ConjunctionAlerts({ pairs, visible, onClose, onSelectPair }: ConjunctionAlertsProps) {
  // Fetch forecast data for featured calls + track record
  const [forecast, setForecast] = useState<any>(null);
  useEffect(() => {
    if (!visible) return;
    fetch('./cdm_forecast.json').then(r => r.ok ? r.json() : null).then(setForecast).catch(() => setForecast(null));
  }, [visible]);

  const tr = forecast?.track_record;
  const featuredCalls: FeaturedCall[] = useMemo(() => forecast?.featured_calls ?? [], [forecast]);
  const escalations = useMemo(() => featuredCalls.filter(c => c.call_type === 'escalation'), [featuredCalls]);
  const safeResolutions = useMemo(() => featuredCalls.filter(c => c.call_type === 'safe'), [featuredCalls]);
  const hasFeatured = featuredCalls.length > 0;

  // Active conjunctions summary
  const activeCount = pairs.length;
  const highCount = pairs.filter(p => riskTier(p) === 'HIGH').length;

  const handleCallClick = useCallback((call: FeaturedCall) => {
    onSelectPair(featuredToScreeningPair(call, pairs));
  }, [onSelectPair, pairs]);

  return (
    <FullPagePanel
      visible={visible}
      onClose={onClose}
      title="Prediction Gallery"
      subtitle="CHA₂DS₂-VASc for satellites — screening for risk escalation, not predicting collisions"
      maxWidth={900}
    >
      {/* Track record summary */}
      {tr && tr.total > 0 && (
        <div style={{ padding: '20px 0', borderBottom: '1px solid #1e1e2c' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
            <div style={{ padding: '12px 14px', background: '#111118', borderRadius: 8, border: '1px solid #1e1e2c' }}>
              <div style={{ fontSize: 22, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: '#e8e8f0' }}>
                {((tr.correct ?? tr.n_correct ?? 0) / tr.total * 100).toFixed(0)}%
              </div>
              <div style={{ fontSize: 12, color: '#55556a', marginTop: 2 }}>Overall Accuracy</div>
              <div style={{ fontSize: 11, color: '#3a3a4a' }}>{tr.correct ?? tr.n_correct}/{tr.total} resolved</div>
            </div>
            <div style={{ padding: '12px 14px', background: '#111118', borderRadius: 8, border: '1px solid #1e1e2c' }}>
              <div style={{ fontSize: 22, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: '#22c55e' }}>
                {tr.tp ?? tr.n_true_positives ?? 0}
              </div>
              <div style={{ fontSize: 12, color: '#55556a', marginTop: 2 }}>Escalations Caught</div>
              <div style={{ fontSize: 11, color: '#3a3a4a' }}>True positives</div>
            </div>
            <div style={{ padding: '12px 14px', background: '#111118', borderRadius: 8, border: '1px solid #1e1e2c' }}>
              <div style={{ fontSize: 22, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: (tr.fn ?? tr.n_false_negatives ?? 0) === 0 ? '#22c55e' : '#ef4444' }}>
                {tr.fn ?? tr.n_false_negatives ?? 0}
              </div>
              <div style={{ fontSize: 12, color: '#55556a', marginTop: 2 }}>Missed Events</div>
              <div style={{ fontSize: 11, color: '#3a3a4a' }}>Zero = never missed</div>
            </div>
            <div style={{ padding: '12px 14px', background: '#111118', borderRadius: 8, border: '1px solid #1e1e2c' }}>
              <div style={{ fontSize: 22, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: '#e8e8f0' }}>
                {activeCount}
              </div>
              <div style={{ fontSize: 12, color: '#55556a', marginTop: 2 }}>Active Pairs</div>
              <div style={{ fontSize: 11, color: '#3a3a4a' }}>{highCount > 0 ? `${highCount} high risk` : 'Monitoring'}</div>
            </div>
          </div>
        </div>
      )}

      {hasFeatured ? (
        <>
          {/* Escalation calls */}
          {escalations.length > 0 && (
            <div style={{ padding: '20px 0' }}>
              <div style={{ fontSize: 13, color: '#ef4444', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: 4 }}>
                Correctly Predicted Escalations
              </div>
              <div style={{ fontSize: 12, color: '#3a3a4a', marginBottom: 14 }}>
                Flagged risk escalation above maneuver threshold — click to view orbital approach on globe
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
                {escalations.map(call => (
                  <FeaturedCallCard key={`${call.sat1_norad}-${call.sat2_norad}-${call.tca}`} call={call} onClick={() => handleCallClick(call)} />
                ))}
              </div>
            </div>
          )}

          {/* Safe resolution calls */}
          {safeResolutions.length > 0 && (
            <div style={{ padding: '20px 0', borderTop: '1px solid #1e1e2c' }}>
              <div style={{ fontSize: 13, color: '#22c55e', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: 4 }}>
                Correctly Predicted Safe Resolutions
              </div>
              <div style={{ fontSize: 12, color: '#3a3a4a', marginBottom: 14 }}>
                Correctly assessed these would not escalate — no unnecessary maneuver
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
                {safeResolutions.map(call => (
                  <FeaturedCallCard key={`${call.sat1_norad}-${call.sat2_norad}-${call.tca}`} call={call} onClick={() => handleCallClick(call)} />
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <div style={{ padding: 48, textAlign: 'center', fontSize: 14, color: '#7c7c96' }}>
          Featured predictions will appear after the next pipeline run resolves conjunction pairs.
        </div>
      )}

      {/* Footer */}
      <div style={{ padding: '16px 0', borderTop: '1px solid #1e1e2c', textAlign: 'center', fontSize: 12, color: '#3a3a4a', lineHeight: 1.5 }}>
        Click any card to view the orbital fly-by on the globe.
        Like CHA₂DS₂-VASc screening for stroke risk — flags when to <em>act</em>, not whether a collision will happen.
      </div>
    </FullPagePanel>
  );
}
