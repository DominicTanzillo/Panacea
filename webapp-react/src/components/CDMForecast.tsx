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

interface ForecastData {
  generated_at: string;
  model: string;
  prediction_task: string;
  n_pairs: number;
  n_actionable: number;
  n_escalating: number;
  model_metrics?: ModelMetrics;
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

function ModelMetricsBanner({ metrics, task }: { metrics: ModelMetrics; task: string }) {
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
            <span className="text-[9px] text-[var(--color-text-muted)] ml-1">F1</span>
          </div>
          <div>
            <span className="text-sm font-bold text-[#8b5cf6]">{metrics.test_auc_pr.toFixed(3)}</span>
            <span className="text-[9px] text-[var(--color-text-muted)] ml-1">AUC-PR</span>
          </div>
          <div className="text-[9px] text-[var(--color-text-muted)]">
            {metrics.n_training_pairs} train / {metrics.n_test_pairs} test pairs
          </div>
        </div>
      )}
    </div>
  );
}

function PairCard({ pair, expanded, onToggle }: { pair: ForecastPair; expanded: boolean; onToggle: () => void }) {
  const level = riskLevel(pair);
  const color = RISK_COLORS[level];

  // Chart data: use hours-to-TCA as X axis (descending = time moves right toward TCA)
  const chartData = useMemo(() => {
    if (!expanded) return [];
    return pair.time_series
      .slice()
      .sort((a, b) => b.time_to_tca_hours - a.time_to_tca_hours) // most distant first
      .map(u => ({
        hoursToTCA: Math.round(u.time_to_tca_hours),
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
          <div className="text-[11px] font-medium truncate">
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
              <div className="text-[var(--color-text-muted)]">P(Exceed 5e-4)</div>
              <div className="font-mono font-semibold" style={{ color: (pair.exceedance_probability ?? 0) >= 0.5 ? '#ff4f5a' : '#4fff8a' }}>
                {formatPct(pair.exceedance_probability ?? 0)}
              </div>
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
                    reversed
                    tick={{ fontSize: 9, fill: '#888' }}
                    label={{ value: 'Hours to TCA \u2192', position: 'insideBottomRight', offset: -2, fill: '#666', fontSize: 9 }}
                  />
                  <YAxis
                    tick={{ fontSize: 9, fill: '#888' }}
                    domain={['auto', 'auto']}
                    tickFormatter={(v: number) => v.toFixed(1)}
                  />
                  <Tooltip
                    contentStyle={{ background: '#1a1a2e', border: '1px solid #444', borderRadius: 8, fontSize: 11, padding: '6px 10px' }}
                    labelFormatter={(v) => `${v}h to TCA`}
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

export function CDMForecast({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  const [data, setData] = useState<ForecastData | null>(null);
  const [filter, setFilter] = useState<'all' | 'critical' | 'high' | 'moderate'>('all');
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  useEffect(() => {
    if (!visible) return;
    fetch('./cdm_forecast.json')
      .then(r => r.ok ? r.json() : null)
      .then(setData)
      .catch(() => setData(null));
  }, [visible]);

  const counts = useMemo(() => {
    return {
      critical: activePairs.filter(p => riskLevel(p) === 'critical').length,
      high: activePairs.filter(p => riskLevel(p) === 'high').length,
      moderate: activePairs.filter(p => riskLevel(p) === 'moderate').length,
    };
  }, [activePairs]);

  // Filter out expired events (TCA in the past)
  const activePairs = useMemo(() => {
    if (!data) return [];
    const now = new Date().toISOString();
    return data.pairs.filter(p => !p.tca || p.tca > now);
  }, [data]);

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
              {data.n_pairs} conjunction pairs tracked
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
            <ModelMetricsBanner metrics={data.model_metrics} task={data.prediction_task} />
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

          {/* Pair list — compact, expandable */}
          <div className="overflow-y-auto flex-1 p-2 space-y-1">
            {filtered.length === 0 ? (
              <div className="text-xs text-[var(--color-text-muted)] text-center py-4">
                No pairs in this category.
              </div>
            ) : (
              filtered.map((pair, i) => (
                <PairCard
                  key={`${pair.sat1_norad}-${pair.sat2_norad}`}
                  pair={pair}
                  expanded={expandedIdx === i}
                  onToggle={() => setExpandedIdx(expandedIdx === i ? null : i)}
                />
              ))
            )}
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
