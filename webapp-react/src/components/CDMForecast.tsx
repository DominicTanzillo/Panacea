import { useState, useEffect, useMemo } from 'react';
import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Area, AreaChart,
} from 'recharts';
import { FullPagePanel } from './Overlay';

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

type RiskLevel = 'critical' | 'high' | 'moderate';

function riskLevel(pair: ForecastPair): RiskLevel {
  if (pair.current_pc >= 5e-3) return 'critical';   // Pc already very high
  if (pair.current_pc >= 5e-4) return 'high';       // Already above maneuver threshold
  return 'moderate';                                  // Below threshold but tracked
}

const RISK_COLORS: Record<RiskLevel, string> = {
  critical: '#ef4444',
  high: '#f59e0b',
  moderate: '#3b82f6',
};

const RISK_LABELS: Record<RiskLevel, string> = {
  critical: 'Critical',
  high: 'High',
  moderate: 'Monitor',
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
  const color = RISK_COLORS[level];
  const isResolved = pair.tca ? pair.tca < new Date().toISOString() : false;
  const ep = pair.ensemble_probability ?? pair.exceedance_probability ?? 0;

  const chartData = useMemo(() => {
    if (!expanded) return [];
    const sorted = pair.time_series.slice().sort((a, b) => b.time_to_tca_hours - a.time_to_tca_hours);
    const seen = new Set<number>();
    return sorted.filter(u => { const h = -Math.round(u.time_to_tca_hours); if (seen.has(h)) return false; seen.add(h); return true; })
      .map(u => ({ hoursToTCA: -Math.round(u.time_to_tca_hours), 'log10(Pc)': u.log10_pc }));
  }, [pair.time_series, expanded]);

  // Build one-line assessment
  const assessment = (() => {
    if (pair.current_pc >= 5e-3) return 'Collision probability is extremely elevated. Maneuver recommended.';
    if (pair.current_pc >= 5e-4) return 'Pc exceeds the maneuver planning threshold.';
    if (ep >= 0.5) return 'Models predict likely escalation above threshold.';
    if (pair.risk_direction === 'de-escalating') return 'Pc is decreasing — conjunction may resolve safely.';
    return 'Below action threshold. Being monitored.';
  })();

  const trendColor = pair.risk_direction === 'escalating' ? '#ef4444' : pair.risk_direction === 'de-escalating' ? '#22c55e' : '#7c7c96';

  return (
    <div style={{ background: '#111118', borderRadius: 8, border: '1px solid #1e1e2c', overflow: 'hidden' }}>
      {/* Collapsed row */}
      <button
        onClick={onToggle}
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
          <span style={{ fontSize: 12, fontWeight: 700, color, background: `${color}12`, padding: '2px 8px', borderRadius: 6 }}>{RISK_LABELS[level]}</span>
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
                  <p style={{ fontSize: 12, color: '#55556a', marginTop: 4 }}>Only {pair.n_updates} CDM update{pair.n_updates > 1 ? 's' : ''} — confidence will improve with more data.</p>
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
                  <MetricRow label="BiLSTM Max Pc" value={`10^${pair.predicted_max_log10_pc.toFixed(1)}`} color={pair.predicted_max_log10_pc > -3.3 ? '#ef4444' : '#22c55e'} />
                )}
                {pair.lstm_escalation_prob != null && (
                  <MetricRow label="LSTM P(Escalation)" value={formatPct(pair.lstm_escalation_prob)} color={(pair.lstm_escalation_prob) >= 0.5 ? '#ef4444' : '#22c55e'} />
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
                <div style={{ fontSize: 12, color: '#55556a', marginBottom: 8 }}>Pc Evolution (log scale)</div>
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
                    <YAxis tick={{ fontSize: 12, fill: '#55556a' }} domain={['auto', 'auto']} tickFormatter={(v: number) => v.toFixed(1)} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: '#111118', border: '1px solid #2a2a3a', borderRadius: 6, fontSize: 12 }} labelFormatter={(v) => `TCA ${v}h`} formatter={(v: unknown) => [`${Number(v).toFixed(2)}`, 'log\u2081\u2080(Pc)']} />
                    <ReferenceLine y={-3.3} stroke="#ef4444" strokeDasharray="4 3" strokeWidth={1.5} strokeOpacity={0.5} />
                    {pair.predicted_max_log10_pc != null && (
                      <ReferenceLine y={pair.predicted_max_log10_pc} stroke="#f59e0b" strokeDasharray="2 3" strokeWidth={1} label={{ value: 'Pred Max', position: 'right', fill: '#f59e0b', fontSize: 12 }} />
                    )}
                    <Area type="monotone" dataKey="log10(Pc)" stroke={color} strokeWidth={2} fill={`url(#pcG-${pair.sat1_norad})`} dot={{ r: 3, fill: color, strokeWidth: 0 }} activeDot={{ r: 5, fill: color }} />
                  </AreaChart>
                </ResponsiveContainer>
                <div style={{ fontSize: 12, color: '#3a3a4a', textAlign: 'center', marginTop: 4 }}>
                  Red line = maneuver threshold (5e-4)
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
  const [filter, setFilter] = useState<'all' | 'critical' | 'high' | 'moderate'>('all');
  const [expandedIdx, setExpandedIdx] = useState<string | null>(null);
  const [_showResolved, _setShowResolved] = useState(false);
  void _showResolved; void _setShowResolved;

  useEffect(() => {
    if (!visible) return;
    fetch('./cdm_forecast.json')
      .then(r => r.ok ? r.json() : null)
      .then(setData)
      .catch(() => setData(null));
  }, [visible]);

  // Show all pairs — active (future TCA) first, then resolved (past TCA).
  // Pipeline now prioritizes future-TCA pairs in the export, but we keep
  // resolved ones visible too so the dashboard isn't empty between events.
  const activePairs = useMemo(() => {
    if (!data) return [];
    const now = new Date().toISOString();
    const future = data.pairs.filter(p => !p.tca || p.tca > now);
    const past = data.pairs.filter(p => p.tca && p.tca <= now);
    return [...future, ...past];
  }, [data]);

  const counts = useMemo(() => {
    return {
      critical: activePairs.filter(p => riskLevel(p) === 'critical').length,
      high: activePairs.filter(p => riskLevel(p) === 'high').length,
      moderate: activePairs.filter(p => riskLevel(p) === 'moderate').length,
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
              The ensemble model monitors {data.n_pairs} active conjunction pairs daily. For each pair, it
              predicts whether collision probability will escalate above the maneuver planning threshold,
              giving operators early warning to plan avoidance.
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
            {(['all', 'critical', 'high', 'moderate'] as const).map(f => {
              const count = f === 'all' ? activePairs.length : counts[f];
              const isActive = filter === f;
              return (
                <button key={f} onClick={() => { setFilter(f); setExpandedIdx(null); }}
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
                  {f === 'all' ? 'All' : f === 'critical' ? 'Critical' : f === 'high' ? 'High' : 'Monitor'} ({count})
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
            <span style={{ textAlign: 'right' }}>P(Exceed)</span>
            <span style={{ textAlign: 'right' }}>Trend</span>
            <span style={{ textAlign: 'right' }}>TCA</span>
            <span style={{ textAlign: 'center' }}>Risk</span>
            <span />
          </div>

          {/* ── Pair list ──────────────────────────────────── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {filtered.map(pair => {
              const key = `${pair.sat1_norad}-${pair.sat2_norad}`;
              return (
                <PairCard key={key} pair={pair} expanded={expandedIdx === key}
                  onToggle={() => setExpandedIdx(expandedIdx === key ? null : key)} />
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
