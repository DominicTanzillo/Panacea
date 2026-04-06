import { useState, useEffect, useMemo } from 'react';
import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Area, AreaChart,
} from 'recharts';
import { FullPagePanel } from './Overlay';
import { GlossaryTooltip } from './Glossary';

interface CDMUpdate {
  update_idx: number;
  log10_pc: number;
  pc: number;
  miss_distance_km: number;
  time_to_tca_hours: number;
}

interface ForecastPair {
  sat1_name: string;
  sat2_name: string;
  sat1_norad: number;
  sat2_norad: number;
  tca: string;
  current_pc: number;
  forecast_pc: number;
  exceedance_probability?: number;
  current_miss_km: number;
  forecast_miss_km: number;
  risk_direction: 'escalating' | 'stable' | 'de-escalating' | 'unknown';
  pc_trend: number;
  confidence: number;
  action_recommended: boolean;
  n_updates: number;
  time_series: CDMUpdate[];
  // CDMSequenceModel regression fields
  predicted_max_pc?: number | null;
  predicted_max_log10_pc?: number | null;
  predicted_min_miss_km?: number | null;
  lstm_escalation_prob?: number | null;
  attention_weights?: number[] | null;
  // Ensemble fields
  ensemble_probability?: number | null;
  exceedance_upper?: number | null;
  exceedance_lower?: number | null;
  // Autoregressive forecast
  forecast_steps?: {
    step: number;
    predicted_log10_pc: number;
    predicted_pc: number;
    predicted_miss_km: number;
    uncertainty_log10_pc: number;
    hours_ahead: number;
  }[];
  // AI summary
  ai_summary?: string;
}

interface ModelMetrics {
  mode: string;
  n_training_pairs: number;
  n_test_pairs: number;
  n_total_pairs: number;
  positive_rate: number;
  test_accuracy: number;
  test_precision: number;
  test_recall: number;
  test_f1: number;
  test_auc_pr: number;
}

interface RegressionMetrics {
  mae_log10_pc: number;
  rmse_log10_pc: number;
  mae_log10_miss: number;
  correlation_pc: number;
  esc_f1: number;
  ece?: number;
  temperature?: number;
  n: number;
}

interface TrackRecordExample {
  sat1_name: string;
  sat2_name: string;
  predicted: string;
  actual: string;
  correct: boolean;
}

interface TrackRecord {
  correct: number;
  total: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  examples?: TrackRecordExample[];
}

interface ConformalMetrics {
  alpha: number;
  regression_coverage: number;
  classification_coverage: number;
  interval_width_log10pc: number;
  q_hat: number;
  n_calibration: number;
}

interface ForecastData {
  generated_at: string;
  model: string;
  prediction_task: string;
  n_pairs: number;
  n_actionable: number;
  n_escalating: number;
  model_metrics?: ModelMetrics;
  conformal?: ConformalMetrics;
  regression_metrics?: RegressionMetrics;
  track_record?: TrackRecord;
  pairs: ForecastPair[];
}

type RiskLevel = 'HIGH' | 'MODERATE' | 'LOW';

function riskLevel(pair: ForecastPair): RiskLevel {
  if (pair.current_pc >= 5e-3) return 'HIGH';        // Pc already very high
  if (pair.current_pc >= 5e-4) return 'MODERATE';    // Already above maneuver threshold
  return 'LOW';                                       // Below threshold but tracked
}

const RISK_COLORS: Record<RiskLevel, string> = {
  HIGH: '#ef4444',
  MODERATE: '#f59e0b',
  LOW: '#22c55e',
};

const RISK_LABELS: Record<RiskLevel, string> = {
  HIGH: '\u26A0 High',
  MODERATE: '-- Moderate',
  LOW: '\u2713 Low',
};

function formatPc(pc: number): string {
  if (!pc || pc <= 0) return '--';
  if (pc >= 0.01) return `${(pc * 100).toFixed(1)}%`;
  if (pc >= 1e-4) {
    const ratio = Math.round(1 / pc);
    return `1 in ${ratio.toLocaleString()}`;
  }
  return pc.toExponential(1);
}

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function shortName(name: string): string {
  // Trim common suffixes for compact display
  return name.replace(/ DEB$/, '').replace(/ R\/B$/, ' R/B').slice(0, 22);
}


function PairCard({ pair, expanded, onToggle }: { pair: ForecastPair; expanded: boolean; onToggle: () => void }) {
  const level = riskLevel(pair);
  const isResolved = pair.tca ? pair.tca < new Date().toISOString() : false;
  const color = isResolved ? '#55556a' : RISK_COLORS[level];
  const ep = pair.ensemble_probability ?? pair.exceedance_probability ?? 0;
  // Chart Y-axis: standardized view shows both thresholds; detail view auto-scales
  const [detailView, setDetailView] = useState(false);

  const chartData = useMemo(() => {
    if (!expanded) return [];
    const sorted = pair.time_series.slice().sort((a, b) => b.time_to_tca_hours - a.time_to_tca_hours);
    const seen = new Set<number>();
    type ChartPoint = { hoursToTCA: number; 'log10(Pc)': number | undefined; forecast: number | undefined; forecastUpper?: number; forecastLower?: number };
    const points: ChartPoint[] = sorted
      .filter(u => { const h = -Math.round(u.time_to_tca_hours); if (seen.has(h)) return false; seen.add(h); return true; })
      .map(u => ({ hoursToTCA: -Math.round(u.time_to_tca_hours), 'log10(Pc)': u.log10_pc, forecast: undefined as number | undefined }));

    if (points.length > 0 && points[points.length - 1].hoursToTCA < 0) {
      const lastPoint = points[points.length - 1];
      // Bridge: set forecast on last real point so dashed line connects
      lastPoint.forecast = lastPoint['log10(Pc)'];

      // Prefer autoregressive forecast_steps (real model predictions with uncertainty)
      if (pair.forecast_steps && pair.forecast_steps.length > 0) {
        const lastHour = lastPoint.hoursToTCA;
        const stepSpan = Math.abs(lastHour) / (pair.forecast_steps.length + 1);
        for (let i = 0; i < pair.forecast_steps.length; i++) {
          const fs = pair.forecast_steps[i];
          // Distribute forecast steps evenly between last CDM and TCA
          const h = Math.round(lastHour + (i + 1) * stepSpan);
          const unc = fs.uncertainty_log10_pc || 0;
          points.push({
            hoursToTCA: h,
            'log10(Pc)': undefined,
            forecast: fs.predicted_log10_pc,
            forecastUpper: fs.predicted_log10_pc + unc,
            forecastLower: fs.predicted_log10_pc - unc,
          });
        }
        // Final point at TCA (0h) using last forecast step
        const lastFs = pair.forecast_steps[pair.forecast_steps.length - 1];
        const lastUnc = lastFs.uncertainty_log10_pc || 0;
        points.push({
          hoursToTCA: 0,
          'log10(Pc)': undefined,
          forecast: lastFs.predicted_log10_pc,
          forecastUpper: lastFs.predicted_log10_pc + lastUnc * 1.2,
          forecastLower: lastFs.predicted_log10_pc - lastUnc * 1.2,
        });
      } else if (pair.forecast_pc > 0) {
        // Fallback: simple interpolation when no autoregressive forecast available
        const startHour = lastPoint.hoursToTCA;
        const startPc = lastPoint['log10(Pc)']!;
        const endPc = Math.log10(Math.max(pair.forecast_pc, 1e-20));
        const span = Math.abs(startHour);
        const nSteps = Math.max(2, Math.min(Math.ceil(span / 2), 12));
        for (let i = 1; i <= nSteps; i++) {
          const t = i / nSteps;
          const h = Math.round(startHour + t * (0 - startHour));
          const ease = t * t;
          const pc = startPc + ease * (endPc - startPc);
          points.push({ hoursToTCA: h, 'log10(Pc)': undefined, forecast: pc });
        }
      }
    }
    return points;
  }, [pair.time_series, pair.forecast_pc, pair.forecast_steps, expanded]);

  // Build one-line assessment — clarify whether risk is current or predicted
  const currentlyAbove = pair.current_pc >= 5e-4;
  // Use autoregressive forecast endpoint when available (matches chart line),
  // otherwise fall back to LogReg forecast_pc
  const effectiveForecastPc = pair.forecast_steps?.length
    ? 10 ** pair.forecast_steps[pair.forecast_steps.length - 1].predicted_log10_pc
    : pair.forecast_pc;
  const forecastAbove = effectiveForecastPc >= 5e-4;
  const assessment = (() => {
    if (isResolved) {
      if (currentlyAbove)
        return 'TCA has passed. Final Pc was above maneuver threshold — this conjunction required action.';
      return 'TCA has passed. Conjunction resolved without exceeding maneuver threshold.';
    }
    if (currentlyAbove && forecastAbove)
      return 'Pc is currently above maneuver threshold and predicted to remain elevated at TCA. Maneuver recommended.';
    if (currentlyAbove && !forecastAbove)
      return 'Pc currently above threshold but forecast to de-escalate before TCA. Continue monitoring.';
    if (!currentlyAbove && ep >= 0.5)
      return `Models predict ${(ep * 100).toFixed(0)}% chance Pc will exceed the maneuver threshold before TCA (not there yet).`;
    if (pair.risk_direction === 'de-escalating')
      return 'Pc is decreasing. Conjunction trending toward safe resolution.';
    return 'Below action threshold. Being monitored.';
  })();

  const trendColor = pair.risk_direction === 'escalating' ? '#ef4444' : pair.risk_direction === 'de-escalating' ? '#22c55e' : '#7c7c96';

  return (
    <div style={{ background: '#111118', borderRadius: 8, border: '1px solid #1e1e2c', overflow: 'hidden' }}>
      {/* Collapsed row */}
      <button
        onClick={onToggle}
        aria-expanded={expanded}
        aria-label={`${expanded ? 'Collapse' : 'Expand'} forecast: ${pair.sat1_name} vs ${pair.sat2_name}, risk level ${level}`}
        style={{
          width: '100%', display: 'grid',
          gridTemplateColumns: '4px 1fr 90px 90px 80px 80px 64px 24px',
          gap: 8, alignItems: 'center', padding: '10px 16px',
          background: 'transparent', border: 'none', cursor: 'pointer',
          textAlign: 'left', fontFamily: 'inherit', transition: 'background 100ms',
        }}
        onMouseEnter={e => { (e.currentTarget).style.background = '#0e0e14'; }}
        onMouseLeave={e => { (e.currentTarget).style.background = 'transparent'; }}
      >
        <div style={{ width: 4, height: 28, borderRadius: 2, background: color }} />
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 13, fontWeight: 500, color: isResolved ? '#55556a' : '#e8e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {isResolved && <span style={{ color: '#55556a', marginRight: 4 }}>PAST</span>}
            {shortName(pair.sat1_name)} <span style={{ color: '#3a3a4a' }}>vs</span> {shortName(pair.sat2_name)}
          </div>
        </div>
        <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: 13, fontWeight: 600, color }}>{formatPc(pair.current_pc)}</div>
        <div style={{ textAlign: 'right', fontFamily: "'JetBrains Mono', monospace", fontSize: 13, color: ep >= 0.5 ? '#ef4444' : '#7c7c96' }}>{formatPct(ep)}</div>
        <div style={{ textAlign: 'right', fontSize: 13, fontWeight: 500, color: trendColor, textTransform: 'capitalize' }}>
          {pair.risk_direction === 'de-escalating' ? 'De-esc' : pair.risk_direction === 'unknown' ? '--' : pair.risk_direction}
        </div>
        <div style={{ textAlign: 'right', fontSize: 13, fontFamily: "'JetBrains Mono', monospace", color: '#7c7c96' }}>
          {pair.tca ? pair.tca.slice(5, 10) : '--'}
        </div>
        <div style={{ textAlign: 'center' }}>
          <span style={{ fontSize: 12, fontWeight: 700, color, background: `${color}12`, padding: '2px 8px', borderRadius: 6 }}>{isResolved ? 'Resolved' : RISK_LABELS[level]}</span>
        </div>
        <span style={{ color: '#3a3a4a', fontSize: 12, textAlign: 'center', transition: 'transform 150ms', transform: expanded ? 'rotate(180deg)' : 'rotate(0)' }}>&#9662;</span>
      </button>

      {/* Expanded detail — clean 2-column layout */}
      {expanded && (
        <div style={{ borderTop: '1px solid #1e1e2c', padding: '16px 20px', animation: 'fadeIn 150ms ease-out' }}>
          <div style={{ display: 'grid', gridTemplateColumns: chartData.length >= 2 ? '1fr 1fr' : '1fr', gap: 24 }}>
            {/* Left: Assessment + metrics */}
            <div>
              {/* One-line assessment */}
              <div style={{ padding: '10px 14px', borderLeft: `3px solid ${color}`, background: `${color}06`, borderRadius: '0 6px 6px 0', marginBottom: 16 }}>
                <p style={{ fontSize: 13, color: '#e8e8f0', lineHeight: 1.5 }}>{assessment}</p>
                {pair.n_updates <= 2 && (
                  <p style={{ fontSize: 12, color: '#55556a', marginTop: 4 }}>Only {pair.n_updates} CDM update{pair.n_updates > 1 ? 's' : ''}. Confidence will improve with more data.</p>
                )}
              </div>

              {/* Key metrics grid */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 13 }}>
                <MetricRow label="Current Pc" value={formatPc(pair.current_pc)} color={color} />
                <MetricRow label="Forecast Pc" value={formatPc(pair.forecast_pc)} color={pair.forecast_pc >= 5e-4 ? '#ef4444' : '#22c55e'} />
                <MetricRow label="P(Exceed 5e-4)" value={formatPct(ep)} color={ep >= 0.5 ? '#ef4444' : '#22c55e'} />
                <MetricRow label="Miss Distance" value={`${pair.current_miss_km.toFixed(0)} km`} />
                <MetricRow label="CDM Updates" value={String(pair.n_updates)} />
                <MetricRow label="TCA" value={pair.tca ? pair.tca.slice(0, 16).replace('T', ' ') : '--'} />
                {pair.predicted_max_log10_pc != null && (
                  <MetricRow label="BiLSTM Max Pc (deep learning model's predicted peak collision probability)" value={`10^${pair.predicted_max_log10_pc.toFixed(1)}`} color={pair.predicted_max_log10_pc > -3.3 ? '#ef4444' : '#22c55e'} />
                )}
                {pair.lstm_escalation_prob != null && (
                  <MetricRow label="LSTM P(Escalation) (neural network's estimated chance of threshold breach)" value={formatPct(pair.lstm_escalation_prob)} color={(pair.lstm_escalation_prob) >= 0.5 ? '#ef4444' : '#22c55e'} />
                )}
              </div>

              {/* AI summary if available */}
              {pair.ai_summary && (
                <div style={{ marginTop: 12, padding: '10px 14px', background: '#0c0c12', borderRadius: 6, border: '1px solid #1e1e2c' }}>
                  <div style={{ fontSize: 12, color: '#55556a', fontWeight: 600, marginBottom: 4 }}>AI Summary</div>
                  <p style={{ fontSize: 13, color: '#7c7c96', lineHeight: 1.5 }}>{pair.ai_summary}</p>
                </div>
              )}
            </div>

            {/* Right: Chart */}
            {chartData.length >= 2 && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <div style={{ fontSize: 12, color: '#55556a' }}>Collision Probability Over Time</div>
                  <button
                    onClick={() => setDetailView(!detailView)}
                    style={{ fontSize: 11, color: '#7c8aff', background: 'rgba(124,138,255,0.08)', border: '1px solid rgba(124,138,255,0.2)', borderRadius: 4, padding: '2px 8px', cursor: 'pointer', fontFamily: 'inherit' }}
                  >
                    {detailView ? 'Standardized' : 'Detail'}
                  </button>
                </div>
                <div role="img" aria-label={`Collision probability evolution chart for ${pair.sat1_name} vs ${pair.sat2_name}. Shows log-scale Pc over time leading to closest approach.`}>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={chartData} margin={{ top: 8, right: 12, left: -10, bottom: 4 }}>
                    <defs>
                      <linearGradient id={`pcG-${pair.sat1_norad}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={color} stopOpacity={0.2} />
                        <stop offset="100%" stopColor={color} stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2c" />
                    <XAxis dataKey="hoursToTCA" type="number" domain={['dataMin', 0]} tick={{ fontSize: 12, fill: '#55556a' }} tickFormatter={(v: number) => `${v}h`} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} />
                    <YAxis
                      tick={{ fontSize: 11, fill: '#55556a' }}
                      domain={detailView ? ['auto', 'auto'] : [-5, -1.5]}
                      tickFormatter={(v: number) => {
                        // Show human-readable Pc at key ticks in standardized view
                        if (!detailView) {
                          if (v === -4) return '1e-4';
                          if (v === -3) return '1e-3';
                          if (v === -2) return '1e-2';
                          if (v === -5) return '1e-5';
                        }
                        return v.toFixed(1);
                      }}
                      axisLine={false}
                      tickLine={false}
                      ticks={detailView ? undefined : [-5, -4, -3.3, -3, -2, -1.5]}
                    />
                    <Tooltip
                      contentStyle={{ background: '#111118', border: '1px solid #2a2a3a', borderRadius: 6, fontSize: 12 }}
                      labelFormatter={(v) => `TCA ${v}h`}
                      formatter={(v: unknown) => {
                        const log10 = Number(v);
                        const pc = Math.pow(10, log10);
                        const readable = pc >= 0.01 ? `${(pc*100).toFixed(1)}%` : pc >= 1e-4 ? `1 in ${Math.round(1/pc).toLocaleString()}` : pc.toExponential(1);
                        return [`${log10.toFixed(2)} (${readable})`, 'Pc'];
                      }}
                    />
                    {/* Screening threshold: 1e-4 (all public CDMs are above this) */}
                    <ReferenceLine y={-4.0} stroke="#3b82f6" strokeDasharray="2 4" strokeWidth={1} strokeOpacity={0.4} label={!detailView ? { value: 'Screening (1e-4)', position: 'insideBottomRight', fill: '#3b82f6', fontSize: 10 } : undefined} />
                    {/* Maneuver threshold: 5e-4 */}
                    <ReferenceLine y={-3.3} stroke="#ef4444" strokeDasharray="4 3" strokeWidth={1.5} strokeOpacity={0.6} label={{ value: 'Maneuver (5e-4)', position: 'insideTopRight', fill: '#ef4444', fontSize: 10 }} />
                    {/* Pred Max line removed — forecast curve already shows predicted trajectory.
                       The yellow line above the red threshold was confusing (yellow=moderate but positioned higher=worse). */}
                    <Area type="monotone" dataKey="log10(Pc)" stroke={color} strokeWidth={2} fill={`url(#pcG-${pair.sat1_norad})`} dot={{ r: 3, fill: color, strokeWidth: 0 }} activeDot={{ r: 5, fill: color }} connectNulls={false} />
                    {/* Uncertainty band from autoregressive forecast */}
                    <Area type="monotone" dataKey="forecastUpper" stroke="none" fill={color} fillOpacity={0.08} connectNulls={false} />
                    <Area type="monotone" dataKey="forecastLower" stroke="none" fill="#111118" fillOpacity={0.9} connectNulls={false} />
                    <Area type="monotone" dataKey="forecast" stroke={color} strokeWidth={2} strokeDasharray="4 3" fill="none" dot={{ r: 4, fill: color, strokeWidth: 2, stroke: '#111118' }} connectNulls={false} />
                  </AreaChart>
                </ResponsiveContainer>
                </div>
                <div style={{ display: 'flex', justifyContent: 'center', gap: 16, fontSize: 11, color: '#3a3a4a', marginTop: 4 }}>
                  <span><span style={{ color: '#ef4444' }}>---</span> Maneuver threshold (5e-4)</span>
                  <span><span style={{ color: '#3b82f6' }}>--</span> Screening threshold (1e-4)</span>
                  <span><span style={{ color: color }}>---</span> Observed</span>
                  <span style={{ borderBottom: `1px dashed ${color}` }}>Forecast</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function MetricRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '4px 0' }}>
      <span style={{ fontSize: 13, color: '#55556a' }}>{label}</span>
      <span style={{ fontSize: 13, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace", color: color ?? '#e8e8f0' }}>{value}</span>
    </div>
  );
}

export function CDMForecast({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  const [data, setData] = useState<ForecastData | null>(null);
  const [filter, setFilter] = useState<'all' | 'HIGH' | 'MODERATE' | 'LOW'>('all');
  const [expandedIdx, setExpandedIdx] = useState<string | null>(null);
  const [_showResolved, _setShowResolved] = useState(false);
  void _showResolved; void _setShowResolved;

  useEffect(() => {
    if (!visible) return;
    fetch('./cdm_forecast.json')
      .then(r => r.ok ? r.json() : null)
      .then((raw) => {
        if (!raw) return null;
        // Normalize track_record field names from pipeline (n_correct, n_wrong, ...)
        // to webapp interface (correct, total, tp, fp, fn, tn)
        if (raw.track_record) {
          const tr = raw.track_record;
          raw.track_record = {
            correct: tr.correct ?? tr.n_correct ?? 0,
            total: tr.total ?? (tr.n_correct ?? 0) + (tr.n_wrong ?? 0),
            tp: tr.tp ?? tr.n_true_positives ?? 0,
            fp: tr.fp ?? tr.n_false_positives ?? 0,
            fn: tr.fn ?? tr.n_false_negatives ?? 0,
            tn: tr.tn ?? tr.n_true_negatives ?? 0,
            examples: tr.examples,
          };
        }
        return raw;
      })
      .then(setData)
      .catch(() => setData(null));
  }, [visible]);

  // Future events first, then resolved (past TCA) clearly separated.
  const activePairs = useMemo(() => {
    if (!data) return [];
    const now = new Date().toISOString();
    const future = data.pairs.filter(p => !p.tca || p.tca > now);
    const past = data.pairs.filter(p => p.tca && p.tca <= now);
    return [...future, ...past];
  }, [data]);

  const nFuture = useMemo(() => {
    if (!data) return 0;
    const now = new Date().toISOString();
    return data.pairs.filter(p => !p.tca || p.tca > now).length;
  }, [data]);

  const counts = useMemo(() => {
    return {
      HIGH: activePairs.filter(p => riskLevel(p) === 'HIGH').length,
      MODERATE: activePairs.filter(p => riskLevel(p) === 'MODERATE').length,
      LOW: activePairs.filter(p => riskLevel(p) === 'LOW').length,
    };
  }, [activePairs]);

  const filtered = useMemo(() => {
    if (filter === 'all') return activePairs;
    return activePairs.filter(p => riskLevel(p) === filter);
  }, [activePairs, filter]);

  return (
    <FullPagePanel
      visible={visible}
      onClose={onClose}
      title="Forecast"
      subtitle={data ? `${data.n_pairs} conjunction pairs \u00b7 ${data.generated_at ? data.generated_at.slice(0, 16).replace('T', ' ') + ' UTC' : ''}` : undefined}
      maxWidth={1000}
    >
      {!data ? (
        <div style={{ padding: 48, textAlign: 'center', fontSize: 14, color: '#7c7c96' }}>
          CDM forecast data not available. Run the daily pipeline to generate predictions.
        </div>
      ) : (
        <>
          {/* ── Value proposition ───────────────────────────── */}
          <div style={{ padding: '24px 0 20px', borderBottom: '1px solid #1e1e2c' }}>
            <div style={{ fontSize: 13, color: '#55556a', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: 8 }}>
              What This Shows
            </div>
            <p style={{ fontSize: 15, color: '#e8e8f0', lineHeight: 1.5, maxWidth: 700, marginBottom: 16 }}>
              Each day, this system receives updated collision data (<GlossaryTooltip term="CDM">CDMs</GlossaryTooltip>) on {data.n_pairs} monitored conjunction pairs.
              It predicts which pairs will escalate to dangerous levels (<GlossaryTooltip term="Pc">Pc</GlossaryTooltip> exceeding
              5 &times; 10<sup>-4</sup>, the maneuver planning threshold) before closest approach, giving
              operators advance warning to plan avoidance maneuvers.
            </p>
            {data.track_record && data.track_record.total > 0 && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
                <TrackStatCard
                  value={`${data.track_record.correct}/${data.track_record.total}`}
                  label="Correct Predictions"
                  detail={`${((data.track_record.correct / data.track_record.total) * 100).toFixed(0)}% accuracy`}
                />
                <TrackStatCard value={String(data.track_record.tp)} label="True Positives" detail="Correctly flagged escalations" color="#22c55e" />
                <TrackStatCard value={String(data.track_record.fn)} label="Missed Events" detail="False negatives" color={data.track_record.fn > 5 ? '#ef4444' : '#f59e0b'} />
                <TrackStatCard value={String(data.track_record.fp)} label="False Alarms" detail="Acceptable cost of safety" />
              </div>
            )}
          </div>

          {/* ── Model metrics (compact) ────────────────────── */}
          {data.model_metrics && (
            <div style={{ padding: '16px 0', borderBottom: '1px solid #1e1e2c', display: 'flex', gap: 24, alignItems: 'center', fontSize: 13 }}>
              <span style={{ color: '#55556a', fontWeight: 600 }}>Model:</span>
              <span style={{ color: '#7c7c96' }}>Accuracy <strong style={{ color: '#e8e8f0', fontFamily: "'JetBrains Mono', monospace" }}>{(data.model_metrics.test_accuracy * 100).toFixed(0)}%</strong></span>
              <span style={{ color: '#7c7c96' }}>F1 <strong style={{ color: '#e8e8f0', fontFamily: "'JetBrains Mono', monospace" }}>{data.model_metrics.test_f1.toFixed(3)}</strong></span>
              <span style={{ color: '#7c7c96' }}>AUC-PR <strong style={{ color: '#e8e8f0', fontFamily: "'JetBrains Mono', monospace" }}>{data.model_metrics.test_auc_pr.toFixed(3)}</strong></span>
              {data.conformal && (
                <span style={{ color: '#7c7c96' }}>Conformal <strong style={{ color: '#e8e8f0', fontFamily: "'JetBrains Mono', monospace" }}>{(data.conformal.regression_coverage * 100).toFixed(0)}%</strong> coverage</span>
              )}
              <span style={{ color: '#55556a', marginLeft: 'auto' }}>{data.model_metrics.n_training_pairs} train / {data.model_metrics.n_test_pairs} test</span>
            </div>
          )}

          {/* ── Filters ────────────────────────────────────── */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '16px 0', borderBottom: '1px solid #1e1e2c' }}>
            {(['all', 'HIGH', 'MODERATE', 'LOW'] as const).map(f => {
              const count = f === 'all' ? activePairs.length : counts[f];
              const isActive = filter === f;
              return (
                <button key={f} onClick={() => { setFilter(f); setExpandedIdx(null); }}
                  aria-label={`Filter by ${f === 'all' ? 'all risk levels' : f + ' risk'}: ${count} pairs`}
                  aria-pressed={isActive}
                  style={{
                    padding: '6px 14px',
                    fontSize: 13,
                    fontWeight: isActive ? 600 : 500,
                    fontFamily: 'inherit',
                    color: isActive ? '#e8e8f0' : '#55556a',
                    background: isActive ? 'rgba(59,130,246,0.08)' : 'transparent',
                    border: isActive ? '1px solid rgba(59,130,246,0.2)' : '1px solid #2a2a3a',
                    borderRadius: 6,
                    cursor: 'pointer',
                    transition: 'all 150ms',
                  }}
                >
                  {f === 'all' ? 'All' : RISK_LABELS[f]} ({count})
                </button>
              );
            })}
          </div>

          {/* ── Column headers ─────────────────────────────── */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '4px 1fr 90px 90px 80px 80px 64px 24px',
            gap: 8, padding: '12px 16px', fontSize: 12, fontWeight: 600,
            color: '#3a3a4a', textTransform: 'uppercase', letterSpacing: '0.04em',
          }}>
            <span />
            <span>Conjunction Pair</span>
            <span style={{ textAlign: 'right' }}>Pc</span>
            <span style={{ textAlign: 'right' }} title="Probability this pair will reach dangerous levels (Pc >= 5e-4, the maneuver threshold)">P(Exceed)</span>
            <span style={{ textAlign: 'right' }}>Trend</span>
            <span style={{ textAlign: 'right' }}>TCA</span>
            <span style={{ textAlign: 'center' }}>Risk</span>
            <span />
          </div>

          {/* ── Pair list ──────────────────────────────────── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {filtered.map((pair, i) => {
              const key = `${pair.sat1_norad}-${pair.sat2_norad}`;
              const isResolved = pair.tca ? pair.tca < new Date().toISOString() : false;
              const prevResolved = i > 0 && filtered[i - 1].tca ? filtered[i - 1].tca! < new Date().toISOString() : false;
              const showDivider = isResolved && !prevResolved && i > 0;
              return (
                <div key={key}>
                  {showDivider && (
                    <div style={{ padding: '12px 16px', margin: '8px 0 4px', borderTop: '1px solid #1e1e2c', fontSize: 12, color: '#55556a', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
                      Resolved Predictions ({filtered.length - nFuture} past TCA)
                    </div>
                  )}
                  <PairCard pair={pair} expanded={expandedIdx === key}
                    onToggle={() => setExpandedIdx(expandedIdx === key ? null : key)} />
                </div>
              );
            })}
          </div>
        </>
      )}
    </FullPagePanel>
  );
}

function TrackStatCard({ value, label, detail, color }: { value: string; label: string; detail: string; color?: string }) {
  return (
    <div style={{ padding: '14px 16px', background: '#111118', borderRadius: 8, border: '1px solid #1e1e2c' }}>
      <div style={{ fontSize: 24, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: color ?? '#e8e8f0', lineHeight: 1.2 }}>
        {value}
      </div>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#7c7c96', marginTop: 4 }}>{label}</div>
      <div style={{ fontSize: 12, color: '#55556a', marginTop: 2 }}>{detail}</div>
    </div>
  );
}
