import { useMemo, useState, useEffect } from 'react';
import * as Tabs from '@radix-ui/react-tabs';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { ModelComparisonResult, StalenessResults, PipelineStats } from '../lib/api';
import type { SatellitePosition } from '../lib/types';

interface RiskDashboardProps {
  modelComparison: ModelComparisonResult[] | null;
  experimentResults: StalenessResults | null;
  satellites: SatellitePosition[];
  visible: boolean;
  onClose: () => void;
}

function DensityTab({ satellites }: { satellites: SatellitePosition[] }) {
  const BIN_WIDTH = 50; // km

  const chartData = useMemo(() => {
    const bins: Record<number, number> = {};
    for (const sat of satellites) {
      const bin = Math.round(sat.alt / BIN_WIDTH) * BIN_WIDTH;
      if (bin >= 0 && bin <= 2000) {
        bins[bin] = (bins[bin] || 0) + 1;
      }
    }
    return Object.entries(bins)
      .map(([alt, count]) => ({ altitude: `${alt}`, count }))
      .sort((a, b) => parseInt(a.altitude) - parseInt(b.altitude));
  }, [satellites]);

  return (
    <div className="p-3">
      <div className="text-xs text-[var(--color-text-muted)] mb-2">
        Object density by altitude ({BIN_WIDTH}km bins, LEO only)
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis
            dataKey="altitude"
            tick={{ fill: '#aaa', fontSize: 9 }}
            interval={3}
            label={{ value: 'Altitude (km)', position: 'insideBottom', offset: -2, fill: '#666', fontSize: 10 }}
          />
          <YAxis tick={{ fill: '#aaa', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8 }}
            labelFormatter={(val) => `${val} km`}
          />
          <Bar dataKey="count" fill="#8b5cf6" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function PipelineTab() {
  const [stats, setStats] = useState<PipelineStats | null>(null);

  useEffect(() => {
    fetch('./pipeline_stats.json')
      .then(r => r.ok ? r.json() : null)
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  if (!stats) {
    return (
      <div className="p-4 text-xs text-[var(--color-text-muted)] text-center">
        Pipeline stats not yet generated. Runs daily at 00:00 UTC.
      </div>
    );
  }

  // Fine-tune progress chart
  const finetuneData = stats.finetune_history.map(ft => ({
    date: ft.date.slice(5, 10),
    'Before': parseFloat(ft.pre_auc_pr.toFixed(3)),
    'After': parseFloat(ft.post_auc_pr.toFixed(3)),
    kept: ft.keep_new_model,
  }));

  // Daily pipeline chart (last 14 days) — deduplicate by date, keep last entry per day
  const deduped = Object.values(
    stats.daily_history.reduce<Record<string, typeof stats.daily_history[0]>>((acc, d) => {
      acc[d.date] = d;
      return acc;
    }, {})
  ).sort((a, b) => a.date.localeCompare(b.date));

  const dailyData = deduped.slice(-14).map(d => ({
    date: d.date.slice(5, 10),
    screened: Math.round((d.n_satellites_screened ?? 0) / 1000),
    predictions: d.n_predictions_logged ?? 0,
  }));

  const fm = stats.forecast_model;
  const testMetrics = fm?.test;

  return (
    <div className="p-3 space-y-4">
      {/* Generated timestamp */}
      {stats.generated_at && (
        <div className="text-[10px] text-[var(--color-text-muted)] text-right">
          Last updated: {new Date(stats.generated_at).toLocaleString()}
        </div>
      )}

      {/* Forecast model accuracy */}
      {fm && (
        <div className="rounded-lg bg-[var(--color-surface-2)] p-3 space-y-2">
          <div className="text-xs font-semibold text-[var(--color-text)]">
            Pc Escalation Forecast Model
          </div>
          <div className="text-[10px] text-[var(--color-text-muted)]">
            Task: Predict whether Pc will exceed 5e-4 (maneuver threshold) before TCA
          </div>
          {testMetrics ? (
            <div className="grid grid-cols-4 gap-2 mt-1">
              <div className="text-center">
                <div className="text-sm font-bold text-[#4fff8a]">{(testMetrics.accuracy * 100).toFixed(1)}%</div>
                <div className="text-[9px] text-[var(--color-text-muted)]">Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-bold text-[#4f8aff]">{testMetrics.f1.toFixed(3)}</div>
                <div className="text-[9px] text-[var(--color-text-muted)]">F1</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-bold text-[#8b5cf6]">{testMetrics.auc_pr.toFixed(3)}</div>
                <div className="text-[9px] text-[var(--color-text-muted)]">AUC-PR</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-bold text-[var(--color-text)]">{fm.n_test ?? 0}</div>
                <div className="text-[9px] text-[var(--color-text-muted)]">Test Pairs</div>
              </div>
            </div>
          ) : (
            <div className="text-[10px] text-[var(--color-text-muted)]">
              Mode: {fm.mode} &middot; {fm.n_pairs_total ?? 0} pairs &middot;
              Metrics available when more CDM data accumulates
            </div>
          )}

          {/* Confusion matrix mini-visualization */}
          {testMetrics && testMetrics.tp != null && testMetrics.fp != null && testMetrics.fn != null && testMetrics.tn != null && (
            <div className="mt-2 pt-2 border-t border-[var(--color-border)]/30">
              <div className="text-[9px] text-[var(--color-text-muted)] mb-1 font-semibold">Confusion Matrix</div>
              <div className="grid grid-cols-2 gap-1 max-w-[160px]">
                <div className="rounded p-1.5 text-center text-[10px]" style={{ background: 'rgba(79, 255, 138, 0.12)' }}>
                  <div className="font-bold text-[#4fff8a]">{testMetrics.tp}</div>
                  <div className="text-[8px] text-[var(--color-text-muted)]">TP</div>
                </div>
                <div className="rounded p-1.5 text-center text-[10px]" style={{ background: 'rgba(255, 79, 90, 0.12)' }}>
                  <div className="font-bold text-[#ff4f5a]">{testMetrics.fp}</div>
                  <div className="text-[8px] text-[var(--color-text-muted)]">FP</div>
                </div>
                <div className="rounded p-1.5 text-center text-[10px]" style={{ background: 'rgba(255, 79, 90, 0.12)' }}>
                  <div className="font-bold text-[#ff4f5a]">{testMetrics.fn}</div>
                  <div className="text-[8px] text-[var(--color-text-muted)]">FN</div>
                </div>
                <div className="rounded p-1.5 text-center text-[10px]" style={{ background: 'rgba(79, 255, 138, 0.12)' }}>
                  <div className="font-bold text-[#4fff8a]">{testMetrics.tn}</div>
                  <div className="text-[8px] text-[var(--color-text-muted)]">TN</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* CDM overview */}
      <div className="grid grid-cols-3 gap-2">
        <div className="rounded-lg bg-[var(--color-surface-2)] p-2 text-center">
          <div className="text-lg font-bold text-[#ff4f5a]">{stats.cdm_stats.pc_high}</div>
          <div className="text-[10px] text-[var(--color-text-muted)]">Pc &ge; 5e-4</div>
        </div>
        <div className="rounded-lg bg-[var(--color-surface-2)] p-2 text-center">
          <div className="text-lg font-bold text-[#ffb84f]">{stats.cdm_stats.pc_moderate}</div>
          <div className="text-[10px] text-[var(--color-text-muted)]">1e-4 &le; Pc &lt; 5e-4</div>
        </div>
        <div className="rounded-lg bg-[var(--color-surface-2)] p-2 text-center">
          <div className="text-lg font-bold text-[#8b5cf6]">{stats.cdm_stats.emergency_count}</div>
          <div className="text-[10px] text-[var(--color-text-muted)]">Emergency Reportable</div>
        </div>
      </div>
      <div className="text-[9px] text-[var(--color-text-muted)] text-center">
        {stats.cdm_stats.total_cdms} total CDMs &middot; All public CDMs have Pc &ge; 1e-4 (Space-Track screening threshold)
      </div>

      {/* Fine-tuning progress */}
      {finetuneData.length > 0 && (
        <>
          <div className="text-xs text-[var(--color-text-muted)]">
            Model Fine-Tuning Progress (AUC-PR)
          </div>
          <ResponsiveContainer width="100%" height={140}>
            <BarChart data={finetuneData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="date" tick={{ fill: '#aaa', fontSize: 10 }} />
              <YAxis domain={[0, 1]} tick={{ fill: '#aaa', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8 }}
                labelStyle={{ color: '#fff' }}
              />
              <Bar dataKey="Before" fill="#666" radius={[2, 2, 0, 0]} />
              <Bar dataKey="After" fill="#4f8aff" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div className="text-[10px] text-[var(--color-text-muted)] text-center">
            Latest: {stats.finetune_history.at(-1)?.post_auc_pr.toFixed(3)} AUC-PR
            ({stats.finetune_history.at(-1)?.n_outcomes} outcomes)
          </div>
        </>
      )}

      {/* Daily pipeline activity */}
      {dailyData.length > 0 && (
        <>
          <div className="text-xs text-[var(--color-text-muted)]">
            Daily Pipeline Activity (14 days)
          </div>
          <ResponsiveContainer width="100%" height={120}>
            <BarChart data={dailyData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="date" tick={{ fill: '#aaa', fontSize: 9 }} />
              <YAxis tick={{ fill: '#aaa', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8 }}
                labelStyle={{ color: '#fff' }}
              />
              <Bar dataKey="screened" fill="#4f8aff" radius={[2, 2, 0, 0]} name="Satellites (×1000)" />
              <Bar dataKey="predictions" fill="#4fff8a" radius={[2, 2, 0, 0]} name="Predictions Logged" />
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}

export function RiskDashboard({
  modelComparison: _modelComparison,
  experimentResults: _experimentResults,
  satellites,
  visible,
  onClose,
}: RiskDashboardProps) {
  void _modelComparison; void _experimentResults; // Reserved for future backend integration
  if (!visible) return null;

  const tabs = ['pipeline', 'density'];

  return (
    <div className="absolute bottom-12 left-4 right-4 max-h-[45vh] rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]/95 backdrop-blur-md shadow-2xl z-20 flex flex-col">
      <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--color-border)]">
        <h3 className="font-semibold text-sm">Risk Dashboard</h3>
        <button
          onClick={onClose}
          className="text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors text-lg leading-none"
        >
          x
        </button>
      </div>

      <Tabs.Root defaultValue="pipeline" className="flex-1 overflow-hidden flex flex-col">
        <Tabs.List className="flex gap-6 border-b border-[var(--color-border)] px-4">
          {tabs.map(tab => (
            <Tabs.Trigger
              key={tab}
              value={tab}
              className="px-4 py-2.5 text-sm font-medium text-[var(--color-text-muted)] border-b-2 border-transparent data-[state=active]:text-[var(--color-text)] data-[state=active]:border-[var(--color-accent)] transition-colors capitalize"
            >
              {tab}
            </Tabs.Trigger>
          ))}
        </Tabs.List>

        <div className="flex-1 overflow-y-auto">
          <Tabs.Content value="pipeline">
            <PipelineTab />
          </Tabs.Content>

          <Tabs.Content value="density">
            <DensityTab satellites={satellites} />
          </Tabs.Content>
        </div>
      </Tabs.Root>
    </div>
  );
}
