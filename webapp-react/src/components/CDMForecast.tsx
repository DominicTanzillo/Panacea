import { useState, useEffect, useMemo } from 'react';
import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Area, AreaChart,
} from 'recharts';

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
  critical: '#ff4f5a',
  high: '#ffb84f',
  moderate: '#4f8aff',
};

const RISK_LABELS: Record<RiskLevel, string> = {
  critical: 'Critical',
  high: 'High',
  moderate: 'Monitor',
};

function formatPc(pc: number): string {
  if (pc >= 0.01) return `${(pc * 100).toFixed(1)}%`;
  return pc.toExponential(1);
}

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function shortName(name: string): string {
  // Trim common suffixes for compact display
  return name.replace(/ DEB$/, '').replace(/ R\/B$/, ' R/B').slice(0, 22);
}

function ModelMetricsBanner({ metrics, task, regressionMetrics }: { metrics: ModelMetrics; task: string; regressionMetrics?: RegressionMetrics }) {
  const hasTest = metrics.test_accuracy > 0;

  return (
    <div className="px-4 py-3 border-b border-[var(--color-border)]">
      <div className="text-[11px] text-[var(--color-text-muted)] mb-2">
        <span className="font-semibold text-[var(--color-text)]">Task:</span>{' '}
        {task || 'Predict Pc exceeding maneuver threshold before TCA'}
      </div>
      {hasTest && (
        <div className="flex items-center gap-4 text-center">
          <div>
            <span className="text-sm font-bold text-[#4fff8a]">{formatPct(metrics.test_accuracy)}</span>
            <span className="text-[9px] text-[var(--color-text-muted)] ml-1">Accuracy</span>
          </div>
          <div>
            <span className="text-sm font-bold text-[#4f8aff]">{metrics.test_f1.toFixed(3)}</span>
            <span className="text-[9px] text-[var(--color-text-muted)] ml-1" title="F1 score: harmonic mean of precision and recall (1.0 = perfect)">F1</span>
          </div>
          <div>
            <span className="text-sm font-bold text-[#8b5cf6]">{metrics.test_auc_pr.toFixed(3)}</span>
            <span className="text-[9px] text-[var(--color-text-muted)] ml-1" title="Area under precision-recall curve (higher = better at finding true positives)">AUC-PR</span>
          </div>
          <div className="text-[9px] text-[var(--color-text-muted)]">
            {metrics.n_training_pairs} train / {metrics.n_test_pairs} test pairs
          </div>
        </div>
      )}
      {regressionMetrics && regressionMetrics.n > 0 && (
        <div className="flex items-center gap-4 text-center mt-2 pt-2 border-t border-[var(--color-border)]/30">
          <div className="text-[9px] text-[var(--color-text-muted)] font-semibold">BiLSTM Ensemble</div>
          <div>
            <span className="text-sm font-bold text-[#ff9f43]">{regressionMetrics.mae_log10_pc.toFixed(3)}</span>
            <span className="text-[9px] text-[var(--color-text-muted)] ml-1" title="Mean absolute error in log10(Pc) prediction">MAE(log Pc)</span>
          </div>
          <div>
            <span className="text-sm font-bold text-[#00cec9]">{regressionMetrics.correlation_pc.toFixed(3)}</span>
            <span className="text-[9px] text-[var(--color-text-muted)] ml-1" title="Pearson correlation between predicted and actual max Pc">Corr</span>
          </div>
          {regressionMetrics.ece != null && (
            <div>
              <span className="text-sm font-bold text-[#a29bfe]">{regressionMetrics.ece.toFixed(3)}</span>
              <span className="text-[9px] text-[var(--color-text-muted)] ml-1" title="Expected Calibration Error — lower means predicted probabilities match actual outcomes (T={regressionMetrics.temperature?.toFixed(2)})">ECE</span>
            </div>
          )}
          <div>
            <span className="text-sm font-bold text-[#fd79a8]">{regressionMetrics.esc_f1.toFixed(3)}</span>
            <span className="text-[9px] text-[var(--color-text-muted)] ml-1" title="Escalation classification F1 score">Esc F1</span>
          </div>
        </div>
      )}
    </div>
  );
}

function PairCard({ pair, expanded, onToggle }: { pair: ForecastPair; expanded: boolean; onToggle: () => void }) {
  const level = riskLevel(pair);
  const color = RISK_COLORS[level];
  const isResolved = pair.tca ? pair.tca < new Date().toISOString() : false;

  // Chart data: use hours-to-TCA as X axis (descending = time moves right toward TCA)
  const chartData = useMemo(() => {
    if (!expanded) return [];
    const sorted = pair.time_series
      .slice()
      .sort((a, b) => b.time_to_tca_hours - a.time_to_tca_hours);
    // Deduplicate by rounded hour value
    const seen = new Set<number>();
    return sorted.filter(u => {
      const h = -Math.round(u.time_to_tca_hours);
      if (seen.has(h)) return false;
      seen.add(h);
      return true;
    }).map(u => ({
      hoursToTCA: -Math.round(u.time_to_tca_hours),
      'log10(Pc)': u.log10_pc,
      pc: u.pc,
      miss_km: u.miss_distance_km,
    }));
  }, [pair.time_series, expanded]);

  return (
    <div className="rounded-lg bg-[var(--color-surface-2)] overflow-hidden">
      {/* Compact row — always visible */}
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-[var(--color-surface)]/30 transition-colors"
      >
        {/* Risk badge */}
        <div
          className="w-1.5 h-8 rounded-full shrink-0"
          style={{ background: color }}
        />

        {/* Names */}
        <div className="min-w-0 flex-1">
          <div className="text-[11px] font-medium truncate" style={{ opacity: isResolved ? 0.6 : 1 }}>
            {isResolved && <span className="text-[8px] text-[var(--color-text-muted)] mr-1">[RESOLVED]</span>}
            {shortName(pair.sat1_name)} vs {shortName(pair.sat2_name)}
          </div>
          <div className="text-[9px] text-[var(--color-text-muted)] font-mono">
            {pair.sat1_norad} / {pair.sat2_norad}
          </div>
        </div>

        {/* Key metrics */}
        <div className="flex items-center gap-3 shrink-0 text-[10px]">
          <div className="text-right">
            <div className="text-[var(--color-text-muted)]">Pc</div>
            <div className="font-mono font-semibold" style={{ color }}>{formatPc(pair.current_pc)}</div>
          </div>
          <div className="text-right">
            <div className="text-[var(--color-text-muted)]">Miss</div>
            <div className="font-mono">{pair.current_miss_km.toFixed(0)} km</div>
          </div>
          <div className="text-right">
            <div className="text-[var(--color-text-muted)]">Updates</div>
            <div className="font-mono">{pair.n_updates}</div>
          </div>
          <span
            className="px-1.5 py-0.5 rounded text-[9px] font-bold"
            style={{ background: `${color}15`, color }}
          >
            {RISK_LABELS[level]}
          </span>
          <span className="text-[var(--color-text-muted)] text-sm">
            {expanded ? '\u25B2' : '\u25BC'}
          </span>
        </div>
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-3 pb-3 space-y-2 border-t border-[var(--color-border)]/50">
          {/* Metrics row */}
          <div className="grid grid-cols-4 gap-3 pt-2 text-[10px]">
            <div>
              <div className="text-[var(--color-text-muted)]">Forecast Pc</div>
              <div className="font-mono font-semibold" style={{ color: pair.forecast_pc >= 5e-4 ? '#ff4f5a' : '#4fff8a' }}>
                {formatPc(pair.forecast_pc)}
              </div>
            </div>
            <div>
              <div className="text-[var(--color-text-muted)]" title="Ensemble: 40% LR + 30% BiLSTM + 30% regression signal">
                {pair.ensemble_probability != null ? 'Ensemble P(Exceed)' : 'P(Exceed 5e-4)'}
              </div>
              <div className="font-mono font-semibold" style={{ color: ((pair.ensemble_probability ?? pair.exceedance_probability ?? 0)) >= 0.5 ? '#ff4f5a' : '#4fff8a' }}>
                {formatPct(pair.ensemble_probability ?? pair.exceedance_probability ?? 0)}
              </div>
              {/* Uncertainty bar */}
              {pair.exceedance_lower != null && pair.exceedance_upper != null && (
                <div className="mt-0.5 h-1.5 rounded-full bg-[var(--color-surface)] relative" title={`Uncertainty: ${formatPct(pair.exceedance_lower)} — ${formatPct(pair.exceedance_upper)}`}>
                  <div
                    className="absolute h-full rounded-full"
                    style={{
                      left: `${pair.exceedance_lower * 100}%`,
                      width: `${Math.max(2, (pair.exceedance_upper - pair.exceedance_lower) * 100)}%`,
                      background: 'rgba(139, 92, 246, 0.5)',
                    }}
                  />
                  <div
                    className="absolute h-full w-0.5 rounded-full bg-white"
                    style={{ left: `${(pair.ensemble_probability ?? 0) * 100}%` }}
                  />
                </div>
              )}
            </div>
            <div>
              <div className="text-[var(--color-text-muted)]">Trend</div>
              <div className="font-semibold capitalize" style={{ color: pair.risk_direction === 'escalating' ? '#ff4f5a' : pair.risk_direction === 'de-escalating' ? '#4fff8a' : '#ffb84f' }}>
                {pair.risk_direction === 'de-escalating' ? 'De-escalating' : pair.risk_direction}
              </div>
            </div>
            <div>
              <div className="text-[var(--color-text-muted)]">TCA</div>
              <div className="font-mono">
                {pair.tca ? pair.tca.slice(5, 16).replace('T', ' ') : '—'}
              </div>
            </div>
          </div>

          {/* Risk Factors — model explainability */}
          {(() => {
            const factors: string[] = [];
            if ((pair.exceedance_probability ?? 0) > 0.7) factors.push('High probability of exceeding maneuver threshold');
            if (pair.pc_trend > 0.05) factors.push('Pc is escalating across recent CDM updates');
            if (pair.pc_trend < -0.05) factors.push('Pc is de-escalating \u2014 risk may be decreasing');
            if (pair.n_updates <= 2) factors.push('Limited CDM data \u2014 prediction confidence is low');
            if (pair.predicted_max_log10_pc != null && pair.predicted_max_log10_pc > -3.3) factors.push(`LSTM model predicts max Pc of ${pair.predicted_max_log10_pc.toFixed(2)} (above 5e-4 threshold)`);
            if (pair.lstm_escalation_prob != null && pair.lstm_escalation_prob > 0.5) factors.push(`Deep learning model: ${(pair.lstm_escalation_prob * 100).toFixed(0)}% escalation probability`);
            if (factors.length === 0) return null;
            return (
              <div
                className="mt-2 pl-2 py-1"
                style={{ borderLeft: `3px solid ${color}`, fontSize: '10px' }}
              >
                <div className="font-semibold text-[var(--color-text-muted)] mb-0.5">Why this risk level?</div>
                {factors.map((f, i) => (
                  <div key={i} className="text-[var(--color-text-muted)] leading-relaxed">{f}</div>
                ))}
              </div>
            );
          })()}

          {/* Regression predictions from CDMSequenceModel */}
          {pair.predicted_max_pc != null && (
            <div className="grid grid-cols-4 gap-3 text-[10px] pt-1 border-t border-[var(--color-border)]/30">
              <div>
                <div className="text-[var(--color-text-muted)]">Pred Max Pc</div>
                <div className="font-mono font-semibold text-[#ff9f43]">
                  {formatPc(pair.predicted_max_pc)}
                </div>
              </div>
              <div>
                <div className="text-[var(--color-text-muted)]">Pred Min Miss</div>
                <div className="font-mono font-semibold text-[#00cec9]">
                  {pair.predicted_min_miss_km != null ? `${pair.predicted_min_miss_km.toFixed(1)} km` : '--'}
                </div>
              </div>
              <div>
                <div className="text-[var(--color-text-muted)]">LSTM P(Esc)</div>
                <div className="font-mono font-semibold" style={{ color: (pair.lstm_escalation_prob ?? 0) >= 0.5 ? '#ff4f5a' : '#4fff8a' }}>
                  {pair.lstm_escalation_prob != null ? formatPct(pair.lstm_escalation_prob) : '--'}
                </div>
              </div>
              <div>
                <div className="text-[var(--color-text-muted)]">log10(Max Pc)</div>
                <div className="font-mono text-[#ff9f43]">
                  {pair.predicted_max_log10_pc != null ? pair.predicted_max_log10_pc.toFixed(2) : '--'}
                </div>
              </div>
            </div>
          )}

          {/* Attention weights — small bar visualization */}
          {pair.attention_weights && pair.attention_weights.length > 0 && (
            <div className="pt-1">
              <div className="text-[9px] text-[var(--color-text-muted)] mb-1">Attention (CDM importance)</div>
              <div className="flex items-end gap-px h-6">
                {pair.attention_weights.map((w, i) => {
                  const maxW = Math.max(...(pair.attention_weights ?? [1]));
                  const pct = maxW > 0 ? (w / maxW) * 100 : 0;
                  return (
                    <div
                      key={i}
                      className="flex-1 rounded-t"
                      style={{
                        height: `${Math.max(pct, 4)}%`,
                        background: `rgba(139, 92, 246, ${0.3 + 0.7 * (w / maxW)})`,
                        minWidth: 2,
                        maxWidth: 12,
                      }}
                      title={`CDM ${i + 1}: ${(w * 100).toFixed(1)}%`}
                    />
                  );
                })}
              </div>
            </div>
          )}

          {/* Pc evolution chart — X axis is hours to TCA */}
          {chartData.length >= 2 && (
            <div className="h-36 mt-1">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 8, right: 12, left: -10, bottom: 4 }}>
                  <defs>
                    <linearGradient id={`pcG-${pair.sat1_norad}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={color} stopOpacity={0.25} />
                      <stop offset="100%" stopColor={color} stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis
                    dataKey="hoursToTCA"
                    type="number"
                    domain={['dataMin', 0]}
                    tick={{ fontSize: 9, fill: '#888' }}
                    tickFormatter={(v: number) => `${v}h`}
                    label={{ value: 'TCA \u2192', position: 'insideBottomRight', offset: -2, fill: '#666', fontSize: 9 }}
                  />
                  <YAxis
                    tick={{ fontSize: 9, fill: '#888' }}
                    domain={['auto', 'auto']}
                    tickFormatter={(v: number) => v.toFixed(1)}
                  />
                  <Tooltip
                    contentStyle={{ background: '#1a1a2e', border: '1px solid #444', borderRadius: 8, fontSize: 11, padding: '6px 10px' }}
                    labelFormatter={(v) => `T${v}h`}
                    formatter={(v: unknown, name?: string) => {
                      if (name === 'log10(Pc)') return [`${Number(v).toFixed(2)}`, 'log\u2081\u2080(Pc)'];
                      return [`${v}`, name ?? ''];
                    }}
                  />
                  <ReferenceLine
                    y={-3.3}
                    stroke="#ff4f5a"
                    strokeDasharray="4 3"
                    strokeWidth={1.5}
                  />
                  {pair.predicted_max_log10_pc != null && (
                    <ReferenceLine
                      y={pair.predicted_max_log10_pc}
                      stroke="#ff9f43"
                      strokeDasharray="2 3"
                      strokeWidth={1.5}
                      label={{ value: 'Pred Max', position: 'right', fill: '#ff9f43', fontSize: 8 }}
                    />
                  )}
                  <Area
                    type="monotone"
                    dataKey="log10(Pc)"
                    stroke={color}
                    strokeWidth={2}
                    fill={`url(#pcG-${pair.sat1_norad})`}
                    dot={{ r: 3, fill: color, strokeWidth: 0 }}
                    activeDot={{ r: 5, fill: color }}
                  />
                </AreaChart>
              </ResponsiveContainer>
              <div className="text-[9px] text-[var(--color-text-muted)] text-center -mt-1">
                Red dashed line = maneuver planning threshold (Pc = 5e-4)
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TrackRecordSection({ record }: { record: TrackRecord }) {
  const accuracy = record.total > 0 ? ((record.correct / record.total) * 100).toFixed(0) : '0';
  return (
    <div className="px-4 py-2 border-b border-[var(--color-border)]">
      <div className="text-[11px] text-[var(--color-text-muted)] mb-1.5">
        Past predictions: {record.correct}/{record.total} correct ({accuracy}% accuracy)
      </div>
      <div className="grid grid-cols-4 gap-1 text-center text-[10px] mb-1.5">
        <div className="rounded p-1" style={{ background: 'rgba(79, 255, 138, 0.12)' }}>
          <div className="font-bold text-[#4fff8a]">{record.tp}</div>
          <div className="text-[var(--color-text-muted)]">TP</div>
        </div>
        <div className="rounded p-1" style={{ background: 'rgba(255, 79, 90, 0.12)' }}>
          <div className="font-bold text-[#ff4f5a]">{record.fp}</div>
          <div className="text-[var(--color-text-muted)]">FP</div>
        </div>
        <div className="rounded p-1" style={{ background: 'rgba(255, 79, 90, 0.12)' }}>
          <div className="font-bold text-[#ff4f5a]">{record.fn}</div>
          <div className="text-[var(--color-text-muted)]">FN</div>
        </div>
        <div className="rounded p-1" style={{ background: 'rgba(79, 255, 138, 0.12)' }}>
          <div className="font-bold text-[#4fff8a]">{record.tn}</div>
          <div className="text-[var(--color-text-muted)]">TN</div>
        </div>
      </div>
      {record.examples && record.examples.length > 0 && (
        <div className="space-y-1">
          {record.examples.slice(0, 3).map((ex, i) => (
            <div key={i} className="text-[9px] text-[var(--color-text-muted)] rounded bg-[var(--color-surface-2)] px-2 py-1 flex items-center justify-between gap-1">
              <span className="truncate">{ex.sat1_name} vs {ex.sat2_name}</span>
              <span className="shrink-0">
                Predicted: {ex.predicted} | Actual: {ex.actual} |{' '}
                <span style={{ color: ex.correct ? '#4fff8a' : '#ff4f5a', fontWeight: 600 }}>
                  {ex.correct ? 'CORRECT' : 'WRONG'}
                </span>
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function CDMForecast({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  const [data, setData] = useState<ForecastData | null>(null);
  const [filter, setFilter] = useState<'all' | 'critical' | 'high' | 'moderate'>('all');
  const [expandedIdx, setExpandedIdx] = useState<string | null>(null);
  const [showResolved, setShowResolved] = useState(false);

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

  if (!visible) return null;

  return (
    <div className="absolute bottom-12 left-4 right-4 max-h-[60vh] rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]/95 backdrop-blur-md shadow-2xl z-20 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--color-border)]">
        <div className="flex items-center gap-3">
          <h3 className="font-semibold text-sm">Pc Escalation Forecast</h3>
          {data && (
            <span className="text-[10px] text-[var(--color-text-muted)]">
              {data.n_pairs} pairs
              {data.generated_at && ` \u00b7 ${data.generated_at.slice(0, 16).replace('T', ' ')} UTC`}
            </span>
          )}
        </div>
        <button
          onClick={onClose}
          className="text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors text-lg leading-none"
        >
          x
        </button>
      </div>

      {!data ? (
        <div className="p-6 text-xs text-[var(--color-text-muted)] text-center">
          CDM forecast data not available. Run the daily pipeline to generate predictions.
        </div>
      ) : (
        <>
          {/* Model metrics */}
          {data.model_metrics && (
            <ModelMetricsBanner metrics={data.model_metrics} task={data.prediction_task} regressionMetrics={data.regression_metrics} />
          )}

          {/* Prediction track record */}
          {data.track_record && (
            <TrackRecordSection record={data.track_record} />
          )}

          {/* Conformal prediction guarantee */}
          {data.conformal && (
            <div className="px-4 py-2 border-b border-[var(--color-border)] flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-[#a29bfe] shrink-0" />
              <div className="text-[10px] text-[var(--color-text-muted)]">
                <span className="font-semibold text-[var(--color-text)]">Conformal Guarantee:</span>{' '}
                {(data.conformal.regression_coverage * 100).toFixed(0)}% of true Pc values fall within prediction intervals
                (target: {((1 - data.conformal.alpha) * 100).toFixed(0)}%, width: {data.conformal.interval_width_log10pc.toFixed(1)} log<sub>10</sub>Pc,
                n={data.conformal.n_calibration} calibration pairs)
              </div>
            </div>
          )}

          {/* Risk breakdown + filter */}
          <div className="flex items-center gap-2 px-4 py-2 border-b border-[var(--color-border)]">
            {(['all', 'critical', 'high', 'moderate'] as const).map(f => {
              const count = f === 'all' ? activePairs.length : counts[f];
              const isActive = filter === f;
              const fColor = f === 'all' ? 'var(--color-text)' : RISK_COLORS[f];
              return (
                <button
                  key={f}
                  onClick={() => { setFilter(f); setExpandedIdx(null); }}
                  className={`px-3 py-1 rounded-full text-[11px] font-medium border transition-all ${
                    isActive
                      ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-text)]'
                      : 'border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
                  }`}
                >
                  <span style={{ color: isActive ? fColor : undefined }}>
                    {f === 'all' ? 'All' : f === 'critical' ? 'Critical' : f === 'high' ? 'High' : 'Monitor'}
                  </span>
                  {' '}({count})
                </button>
              );
            })}
          </div>

          {/* Pair list — split into active and resolved */}
          <div className="overflow-y-auto flex-1 p-2 space-y-1">
            {(() => {
              const now = new Date().toISOString();
              const activePairsList = filtered.filter(p => !p.tca || p.tca > now);
              const resolvedPairsList = filtered.filter(p => p.tca && p.tca <= now);

              return (
                <>
                  {/* Active Conjunctions */}
                  <div className="text-[11px] font-semibold text-[var(--color-text)] px-1 pt-1 pb-0.5">
                    Active Conjunctions ({activePairsList.length})
                  </div>
                  {activePairsList.length === 0 ? (
                    <div className="text-xs text-[var(--color-text-muted)] text-center py-3">
                      No active pairs in this category.
                    </div>
                  ) : (
                    activePairsList.map((pair) => {
                      const key = `active-${pair.sat1_norad}-${pair.sat2_norad}`;
                      return (
                        <PairCard
                          key={key}
                          pair={pair}
                          expanded={expandedIdx === key}
                          onToggle={() => setExpandedIdx(expandedIdx === key ? null : key)}
                        />
                      );
                    })
                  )}

                  {/* Resolved Events */}
                  {resolvedPairsList.length > 0 && (
                    <>
                      <button
                        onClick={() => setShowResolved(!showResolved)}
                        className="w-full flex items-center justify-between text-[11px] font-semibold text-[var(--color-text-muted)] px-1 pt-3 pb-0.5 hover:text-[var(--color-text)] transition-colors"
                      >
                        <span>Resolved Events ({resolvedPairsList.length})</span>
                        <span className="text-[10px]">{showResolved ? '\u25B2' : '\u25BC'}</span>
                      </button>
                      {showResolved && resolvedPairsList.map((pair) => {
                        const key = `resolved-${pair.sat1_norad}-${pair.sat2_norad}`;
                        return (
                          <PairCard
                            key={key}
                            pair={pair}
                            expanded={expandedIdx === key}
                            onToggle={() => setExpandedIdx(expandedIdx === key ? null : key)}
                          />
                        );
                      })}
                    </>
                  )}
                </>
              );
            })()}
          </div>

          {/* Footer */}
          <div className="px-4 py-1.5 border-t border-[var(--color-border)] text-center">
            <span className="text-[9px] text-[var(--color-text-muted)]">
              Retrained daily on resolved pairs &middot; Source: Space-Track 18th SDS CDMs &middot;
              Click a pair to see Pc evolution
            </span>
          </div>
        </>
      )}
    </div>
  );
}
