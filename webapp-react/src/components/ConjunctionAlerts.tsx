import { useState, useEffect, useMemo } from 'react';
import {
  XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer, Area, AreaChart,
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
  if (hours >= 1) return `${hours.toFixed(0)} hours`;
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

function FeaturedCallCard({ call }: { call: FeaturedCall }) {
  const isEscalation = call.call_type === 'escalation';
  const color = isEscalation ? '#ef4444' : '#22c55e';

  const chartData = useMemo(() => {
    const sorted = call.time_series.slice().sort((a, b) => b.time_to_tca_hours - a.time_to_tca_hours);
    return sorted.map(u => ({
      hoursToTCA: -Math.round(u.time_to_tca_hours),
      'log10(Pc)': u.log10_pc,
    }));
  }, [call.time_series]);

  return (
    <div style={{ background: '#111118', borderRadius: 10, border: '1px solid #1e1e2c', overflow: 'hidden' }}>
      {/* Header: names + lead time + outcome */}
      <div style={{ padding: '12px 16px 10px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: '#e8e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, minWidth: 0 }}>
            {call.sat1_name} <span style={{ color: '#3a3a4a', fontWeight: 400 }}>vs</span> {call.sat2_name}
          </div>
          <span style={{ fontSize: 11, fontWeight: 700, color, background: `${color}10`, padding: '2px 8px', borderRadius: 4, whiteSpace: 'nowrap', flexShrink: 0, marginLeft: 8 }}>
            {isEscalation ? 'Escalation' : 'Safe'}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 12, fontSize: 12, color: '#55556a' }}>
          <span><strong style={{ color: '#3b82f6' }}>{formatLeadTime(call.prediction_lead_hours)}</strong> advance warning</span>
          <span>Pc {formatPc(call.current_pc)}</span>
          <span>{call.n_updates} CDMs</span>
        </div>
      </div>

      {/* Pc evolution chart — always present since we require 3+ CDMs */}
      <div style={{ padding: '0 12px 6px' }}>
        <ResponsiveContainer width="100%" height={110}>
          <AreaChart data={chartData} margin={{ top: 4, right: 8, left: -10, bottom: 4 }}>
            <defs>
              <linearGradient id={`fcg-${call.sat1_norad}-${call.sat2_norad}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.15} />
                <stop offset="100%" stopColor={color} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <XAxis dataKey="hoursToTCA" type="number" domain={['dataMin', 0]}
              tick={{ fontSize: 10, fill: '#3a3a4a' }}
              tickFormatter={(v: number) => `${v}h`}
              axisLine={{ stroke: '#1e1e2c' }} tickLine={false} />
            <YAxis tick={{ fontSize: 10, fill: '#3a3a4a' }} domain={[-5, -1.5]}
              tickFormatter={(v: number) => {
                if (v === -4) return '1e-4';
                if (v === -3) return '1e-3';
                if (v === -2) return '1e-2';
                return '';
              }}
              axisLine={false} tickLine={false}
              ticks={[-4, -3.3, -3, -2]} />
            <Tooltip
              contentStyle={{ background: '#111118', border: '1px solid #2a2a3a', borderRadius: 6, fontSize: 11 }}
              labelFormatter={(v) => `TCA ${v}h`}
              formatter={(v: unknown) => {
                const log10 = Number(v);
                const pc = Math.pow(10, log10);
                const readable = pc >= 0.01 ? `${(pc*100).toFixed(1)}%` : pc >= 1e-4 ? `1 in ${Math.round(1/pc).toLocaleString()}` : pc.toExponential(1);
                return [`${log10.toFixed(2)} (${readable})`, 'Pc'];
              }}
            />
            <ReferenceLine y={-3.3} stroke="#ef4444" strokeDasharray="3 3" strokeWidth={1} strokeOpacity={0.5}
              label={{ value: '5e-4', position: 'insideTopRight', fill: '#ef4444', fontSize: 9 }} />
            <Area type="monotone" dataKey="log10(Pc)" stroke={color} strokeWidth={2}
              fill={`url(#fcg-${call.sat1_norad}-${call.sat2_norad})`}
              dot={{ r: 2.5, fill: color, strokeWidth: 0 }} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* One-line outcome */}
      <div style={{ padding: '6px 16px 10px', fontSize: 11, color: '#3a3a4a' }}>
        {isEscalation
          ? <>{(call.ensemble_probability * 100).toFixed(0)}% confidence &middot; Pc peaked at {formatPc(call.current_pc)} &middot; {call.current_miss_km.toFixed(0)} km miss</>
          : <>Peak Pc {formatPc(Math.max(...call.time_series.map(u => u.pc)))} stayed below threshold &middot; {call.current_miss_km.toFixed(0)} km miss</>
        }
      </div>
    </div>
  );
}

export function ConjunctionAlerts({ pairs, visible, onClose, onSelectPair }: ConjunctionAlertsProps) {
  void onSelectPair;  // kept for interface compat

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

  // Fallback: compute featured from track_record examples when featured_calls not yet available
  const hasFeatured = featuredCalls.length > 0;

  // Active conjunctions summary
  const activeCount = pairs.length;
  const highCount = pairs.filter(p => riskTier(p) === 'HIGH').length;

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
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                <div style={{ fontSize: 13, color: '#ef4444', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
                  Correctly Predicted Escalations
                </div>
                <div style={{ fontSize: 12, color: '#3a3a4a' }}>
                  Model flagged these pairs before Pc exceeded the maneuver threshold
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {escalations.map(call => (
                  <FeaturedCallCard key={`${call.sat1_norad}-${call.sat2_norad}-${call.tca}`} call={call} />
                ))}
              </div>
            </div>
          )}

          {/* Safe resolution calls */}
          {safeResolutions.length > 0 && (
            <div style={{ padding: '20px 0', borderTop: '1px solid #1e1e2c' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                <div style={{ fontSize: 13, color: '#22c55e', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
                  Correctly Predicted Safe Resolutions
                </div>
                <div style={{ fontSize: 12, color: '#3a3a4a' }}>
                  Model correctly assessed these would not escalate
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {safeResolutions.map(call => (
                  <FeaturedCallCard key={`${call.sat1_norad}-${call.sat2_norad}-${call.tca}`} call={call} />
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <div style={{ padding: 48, textAlign: 'center', fontSize: 14, color: '#7c7c96' }}>
          Featured predictions will appear after the next pipeline run resolves conjunction pairs.
          The pipeline runs daily at 00:00 UTC and curates the most impressive correct predictions.
        </div>
      )}

      {/* Footer */}
      <div style={{ padding: '16px 0', borderTop: '1px solid #1e1e2c', textAlign: 'center', fontSize: 12, color: '#3a3a4a', lineHeight: 1.5 }}>
        Updated each pipeline run. Like CHA₂DS₂-VASc screening for stroke risk, these predictions flag when
        to <em>act</em> (plan a maneuver) — not whether a collision will happen.
        Escalation = Pc exceeded 5 &times; 10<sup>-4</sup> (start planning). Safe = risk stayed below threshold through closest approach.
      </div>
    </FullPagePanel>
  );
}
