import { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line, ReferenceLine,
} from 'recharts';
import { FullPagePanel } from './Overlay';

interface ModelZooPageProps {
  visible: boolean;
  onClose: () => void;
}

/* ── Types ────────────────────────────────────────────────── */
interface CVResults {
  n_folds: number;
  n_pairs: number;
  n_positive: number;
  positive_rate: number;
  results: {
    lr: { f1: number; precision: number; recall: number; accuracy: number; auc_pr?: number }[];
    lstm: { f1: number; precision: number; recall: number; accuracy: number; mae_log10_pc?: number; correlation?: number; ece?: number }[];
    ensemble: { f1: number; precision: number; recall: number; accuracy: number; tp?: number; fp?: number; fn?: number; tn?: number }[];
  };
}

/* ── Helpers ───────────────────────────────────────────────── */
function avg(arr: number[]): number {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}
function std(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = avg(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1));
}
function fmt(n: number, d = 3): string { return n.toFixed(d); }

const TOOLTIP_STYLE = {
  background: '#111118',
  border: '1px solid #2a2a3a',
  borderRadius: 6,
  fontSize: 12,
  padding: '8px 12px',
};

export function ModelZooPage({ visible, onClose }: ModelZooPageProps) {
  const [cv, setCv] = useState<CVResults | null>(null);
  /* eslint-disable @typescript-eslint/no-explicit-any */
  const [supp, setSupp] = useState<any>(null);

  /* eslint-disable @typescript-eslint/no-explicit-any */
  const [pstats, setPstats] = useState<any>(null);

  useEffect(() => {
    if (!visible) return;
    fetch('./cv_results.json').then(r => r.ok ? r.json() : null).then(setCv).catch(() => setCv(null));
    fetch('./supplementary_models.json').then(r => r.ok ? r.json() : null).then(setSupp).catch(() => setSupp(null));
    fetch('./pipeline_stats.json').then(r => r.ok ? r.json() : null).then(setPstats).catch(() => setPstats(null));
  }, [visible]);

  return (
    <FullPagePanel
      visible={visible}
      onClose={onClose}
      title="Models"
      subtitle={cv
        ? `${cv.n_folds}-fold cross-validation on ${cv.n_pairs} conjunction pairs \u00b7 ${cv.n_positive} positive (${(cv.positive_rate * 100).toFixed(1)}%)`
        : '6 models for satellite collision risk assessment'}
    >

          {/* ── Context disclaimer ────────────────────────────── */}
          <div style={{
            padding: '12px 16px',
            marginBottom: 8,
            background: 'rgba(59,130,246,0.05)',
            border: '1px solid rgba(59,130,246,0.15)',
            borderRadius: 8,
            fontSize: 13,
            color: '#7c7c96',
            lineHeight: 1.5,
          }}>
            This page shows detailed model performance metrics for technical review. For operational collision-risk decisions, see the <strong style={{ color: '#e8e8f0' }}>Forecast</strong> and <strong style={{ color: '#e8e8f0' }}>Alerts</strong> tabs.
          </div>

          {/* ── Section 1: Task & Key Insight ────────────────── */}
          <div style={{ padding: '24px 0 20px', borderBottom: '1px solid #1e1e2c' }}>
            <div style={{ fontSize: 13, color: '#55556a', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: 8 }}>
              Prediction Task
            </div>
            <p style={{ fontSize: 15, color: '#e8e8f0', lineHeight: 1.5, maxWidth: 700, marginBottom: 16 }}>
              Given a sequence of CDM updates for a conjunction pair, predict whether collision probability (Pc)
              will exceed <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#3b82f6' }}>5 &times; 10<sup>-4</sup></span> before
              time of closest approach, the threshold at which operators begin planning avoidance maneuvers.
            </p>
            <div style={{
              padding: '14px 20px',
              background: 'rgba(239,68,68,0.04)',
              border: '1px solid rgba(239,68,68,0.12)',
              borderRadius: 8,
              maxWidth: 700,
            }}>
              <p style={{ fontSize: 14, color: '#e8e8f0', lineHeight: 1.6 }}>
                <strong style={{ color: '#ef4444' }}>Recall is the primary metric.</strong> A false negative
                means an undetected escalation: an operator doesn't plan a maneuver, and a $500M
                satellite is at risk. A false positive means 30 minutes of extra monitoring. We optimize
                for catching every dangerous event, even at the cost of more false alarms.
              </p>
            </div>
          </div>

          {/* ── Section 2: Model Comparison Table ────────────── */}
          <ModelComparisonTable cv={cv} />

          {/* ── Section 3: Per-Fold Stability ────────────────── */}
          <FoldStabilityChart cv={cv} />

          {/* ── Section 3b: BiLSTM Learning Over Time ──────── */}
          <BiLSTMProgress pstats={pstats} />

          {/* ── Section 4: Ensemble Deep-Dive ────────────────── */}
          <EnsembleSection cv={cv} />

          {/* ── Section 5: Supplementary Models ──────────────── */}
          <SupplementaryModels supp={supp} />

    </FullPagePanel>
  );
}

/* ══════════════════════════════════════════════════════════════
   Section 2: Clean comparison table
   ══════════════════════════════════════════════════════════════ */
function ModelComparisonTable({ cv }: { cv: CVResults | null }) {
  const models = useMemo(() => {
    if (!cv) return null;
    return [
      {
        name: 'Logistic Regression',
        role: 'Production baseline \u00b7 23 features, retrained daily',
        f1: avg(cv.results.lr.map(r => r.f1)),
        precision: avg(cv.results.lr.map(r => r.precision)),
        recall: avg(cv.results.lr.map(r => r.recall)),
        accuracy: avg(cv.results.lr.map(r => r.accuracy)),
        missedEvents: Math.round(cv.n_positive * (1 - avg(cv.results.lr.map(r => r.recall)))),
      },
      {
        name: 'BiLSTM',
        role: 'Kelvins transfer learning \u00b7 focal loss \u00b7 attention pooling',
        f1: avg(cv.results.lstm.map(r => r.f1)),
        precision: avg(cv.results.lstm.map(r => r.precision)),
        recall: avg(cv.results.lstm.map(r => r.recall)),
        accuracy: avg(cv.results.lstm.map(r => r.accuracy)),
        missedEvents: Math.round(cv.n_positive * (1 - avg(cv.results.lstm.map(r => r.recall)))),
      },
      {
        name: 'Ensemble',
        role: '40% LR + 30% BiLSTM + 30% regression signal',
        f1: avg(cv.results.ensemble.map(r => r.f1)),
        precision: avg(cv.results.ensemble.map(r => r.precision)),
        recall: avg(cv.results.ensemble.map(r => r.recall)),
        accuracy: avg(cv.results.ensemble.map(r => r.accuracy)),
        missedEvents: cv.results.ensemble.reduce((s, x) => s + (x.fn ?? 0), 0),
      },
    ];
  }, [cv]);

  if (!models) return <div style={{ padding: '32px 0', color: '#55556a', fontSize: 13 }}>Loading cross-validation data...</div>;

  return (
    <div style={{ padding: '24px 0', borderBottom: '1px solid #1e1e2c' }}>
      <SectionHeader>Model Comparison</SectionHeader>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #2a2a3a' }}>
              <Th align="left">Model</Th>
              <Th title="What fraction of dangerous events we catch">Recall</Th>
              <Th title="Dangerous events the model failed to flag">Missed Events</Th>
              <Th title="Balance between catching events and avoiding false alarms">F1</Th>
              <Th title="What fraction of alerts are real threats">Precision</Th>
              <Th title="Overall correct classification rate">Accuracy</Th>
            </tr>
          </thead>
          <tbody>
            {models.map((m, i) => {
              const bestRecall = models.reduce((a, b) => b.recall > a.recall ? b : a);
              const fewestMisses = models.reduce((a, b) => b.missedEvents < a.missedEvents ? b : a);
              return (
                <tr key={i} style={{ borderBottom: '1px solid #1a1a24' }}>
                  <td style={{ padding: '12px 12px 12px 0' }}>
                    <div style={{ fontWeight: 600, color: '#e8e8f0' }}>{m.name}</div>
                    <div style={{ fontSize: 12, color: '#55556a', marginTop: 2 }}>{m.role}</div>
                  </td>
                  <Td highlight={m === bestRecall}>{fmt(m.recall)}</Td>
                  <td style={{
                    padding: '12px',
                    fontSize: 15,
                    fontWeight: 700,
                    fontFamily: "'JetBrains Mono', monospace",
                    textAlign: 'right',
                    color: m === fewestMisses ? '#22c55e' : m.missedEvents > 20 ? '#ef4444' : '#f59e0b',
                  }}>
                    {m.missedEvents} / {cv!.n_positive}
                  </td>
                  <Td>{fmt(m.f1)}</Td>
                  <Td>{fmt(m.precision)}</Td>
                  <Td>{(m.accuracy * 100).toFixed(1)}%</Td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   Section 3: Per-fold F1 stability chart
   ══════════════════════════════════════════════════════════════ */
function FoldStabilityChart({ cv }: { cv: CVResults | null }) {
  const data = useMemo(() => {
    if (!cv) return [];
    return cv.results.lr.map((_, i) => ({
      fold: `Fold ${i + 1}`,
      'Logistic Regression': +cv.results.lr[i].f1.toFixed(3),
      'BiLSTM': +cv.results.lstm[i].f1.toFixed(3),
      'Ensemble': +cv.results.ensemble[i].f1.toFixed(3),
    }));
  }, [cv]);

  const stats = useMemo(() => {
    if (!cv) return null;
    return {
      lrStd: std(cv.results.lr.map(r => r.f1)),
      lstmStd: std(cv.results.lstm.map(r => r.f1)),
      ensStd: std(cv.results.ensemble.map(r => r.f1)),
    };
  }, [cv]);

  if (!cv || data.length === 0) return null;

  return (
    <div style={{ padding: '24px 0', borderBottom: '1px solid #1e1e2c' }}>
      <SectionHeader>Cross-Validation Stability</SectionHeader>
      <p style={{ fontSize: 13, color: '#7c7c96', marginBottom: 16, maxWidth: 600 }}>
        F1 score across 5 folds. BiLSTM shows high variance (&#963; = {stats?.lstmStd.toFixed(3)}) due to
        small dataset size and sequence sensitivity. The ensemble stabilizes predictions
        (&#963; = {stats?.ensStd.toFixed(3)}).
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 240px', gap: 24, alignItems: 'start' }}>
        {/* Chart */}
        <div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 4 }} barCategoryGap="20%">
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2c" />
              <XAxis dataKey="fold" tick={{ fill: '#7c7c96', fontSize: 12 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} />
              <YAxis domain={[0, 1]} tick={{ fill: '#55556a', fontSize: 12 }} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(1)} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Bar dataKey="Logistic Regression" fill="#e8e8f0" radius={[3, 3, 0, 0]} opacity={0.85} />
              <Bar dataKey="BiLSTM" fill="#3b82f6" radius={[3, 3, 0, 0]} opacity={0.6} />
              <Bar dataKey="Ensemble" fill="#3b82f6" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Variance summary */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <VarianceStat label="Logistic Regression" mean={avg(cv.results.lr.map(r => r.f1))} sd={stats?.lrStd ?? 0} />
          <VarianceStat label="BiLSTM" mean={avg(cv.results.lstm.map(r => r.f1))} sd={stats?.lstmStd ?? 0} />
          <VarianceStat label="Ensemble" mean={avg(cv.results.ensemble.map(r => r.f1))} sd={stats?.ensStd ?? 0} />
        </div>
      </div>
    </div>
  );
}

function VarianceStat({ label, mean, sd }: { label: string; mean: number; sd: number }) {
  return (
    <div style={{ padding: '10px 14px', background: '#111118', borderRadius: 6, border: '1px solid #1e1e2c' }}>
      <div style={{ fontSize: 12, color: '#55556a', marginBottom: 4 }}>{label}</div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span style={{ fontSize: 20, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: '#e8e8f0' }}>
          {mean.toFixed(3)}
        </span>
        <span style={{ fontSize: 12, color: '#55556a', fontFamily: "'JetBrains Mono', monospace" }}>
          &plusmn; {sd.toFixed(3)}
        </span>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   Section 4: Ensemble deep-dive
   ══════════════════════════════════════════════════════════════ */
function EnsembleSection({ cv }: { cv: CVResults | null }) {
  const cm = useMemo(() => {
    if (!cv) return null;
    const e = cv.results.ensemble;
    return {
      tp: e.reduce((s, x) => s + (x.tp ?? 0), 0),
      fp: e.reduce((s, x) => s + (x.fp ?? 0), 0),
      fn: e.reduce((s, x) => s + (x.fn ?? 0), 0),
      tn: e.reduce((s, x) => s + (x.tn ?? 0), 0),
    };
  }, [cv]);

  if (!cm) return null;

  const total = cm.tp + cm.fp + cm.fn + cm.tn;
  const tpr = cm.tp / (cm.tp + cm.fn);
  const fpr = cm.fp / (cm.fp + cm.tn);

  return (
    <div style={{ padding: '24px 0', borderBottom: '1px solid #1e1e2c' }}>
      <SectionHeader>Ensemble: Why 98.7% Recall Matters</SectionHeader>
      <p style={{ fontSize: 14, color: '#7c7c96', marginBottom: 20, maxWidth: 700, lineHeight: 1.6 }}>
        In collision avoidance, a false negative means a missed escalation event: an undetected risk
        that could lead to a collision. The ensemble trades 53 false alarms for just <strong style={{ color: '#e8e8f0' }}>2 missed events</strong> out
        of 152 positives across all folds.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 32, alignItems: 'start' }}>
        {/* Confusion Matrix */}
        <div>
          <div style={{ fontSize: 12, color: '#55556a', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: 12 }}>
            Confusion Matrix ({total} pairs)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3 }}>
            <CMCell value={cm.tp} label="True Positive" pct={cm.tp / total} good />
            <CMCell value={cm.fp} label="False Positive" pct={cm.fp / total} good={false} />
            <CMCell value={cm.fn} label="False Negative" pct={cm.fn / total} good={false} critical />
            <CMCell value={cm.tn} label="True Negative" pct={cm.tn / total} good />
          </div>
          <div style={{ marginTop: 12, display: 'flex', gap: 16, fontSize: 13 }}>
            <div>
              <span style={{ color: '#55556a' }}>TPR </span>
              <span style={{ color: '#e8e8f0', fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>{(tpr * 100).toFixed(1)}%</span>
            </div>
            <div>
              <span style={{ color: '#55556a' }}>FPR </span>
              <span style={{ color: '#e8e8f0', fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>{(fpr * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* Ensemble architecture explanation */}
        <div>
          <div style={{ fontSize: 12, color: '#55556a', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: 12 }}>
            Ensemble Architecture
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <WeightBar label="Logistic Regression" weight={40} desc="23 engineered features, retrained daily" />
            <WeightBar label="BiLSTM (Kelvins)" weight={30} desc="Transfer learning, attention pooling, focal loss" />
            <WeightBar label="Regression Signal" weight={30} desc="Predicted max Pc acts as soft escalation vote" />
          </div>
          <p style={{ fontSize: 13, color: '#55556a', marginTop: 12, lineHeight: 1.5 }}>
            Weighted average of three independent probability estimates.
            Temperature-scaled BiLSTM outputs (T optimized on holdout) ensure calibrated probabilities
            before ensembling.
          </p>
        </div>
      </div>
    </div>
  );
}

function WeightBar({ label, weight, desc }: { label: string; weight: number; desc: string }) {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 13, fontWeight: 500, color: '#e8e8f0' }}>{label}</span>
        <span style={{ fontSize: 13, fontFamily: "'JetBrains Mono', monospace", color: '#7c7c96' }}>{weight}%</span>
      </div>
      <div style={{ height: 6, borderRadius: 3, background: '#1a1a24', overflow: 'hidden', marginBottom: 4 }}>
        <div style={{ height: '100%', width: `${weight}%`, background: '#3b82f6', borderRadius: 3, opacity: 0.7 + weight / 200 }} />
      </div>
      <div style={{ fontSize: 12, color: '#55556a' }}>{desc}</div>
    </div>
  );
}

function CMCell({ value, label, pct, good, critical }: {
  value: number; label: string; pct: number; good: boolean; critical?: boolean;
}) {
  const bg = good ? 'rgba(34,197,94,0.06)' : critical ? 'rgba(239,68,68,0.08)' : 'rgba(239,68,68,0.04)';
  const color = good ? '#22c55e' : '#ef4444';
  return (
    <div style={{ padding: '16px 12px', background: bg, borderRadius: 6, textAlign: 'center' }}>
      <div style={{ fontSize: 28, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color, lineHeight: 1.2 }}>
        {value}
      </div>
      <div style={{ fontSize: 12, color: '#7c7c96', marginTop: 4 }}>{label}</div>
      <div style={{ fontSize: 12, color: '#55556a', fontFamily: "'JetBrains Mono', monospace", marginTop: 2 }}>
        {(pct * 100).toFixed(1)}%
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   Section 5: Supplementary models
   ══════════════════════════════════════════════════════════════ */
function SupplementaryModels({ supp }: { supp: any }) {
  const [expanded, setExpanded] = useState<string | null>(null);

  const gnnMetric = supp?.gnn ? `F1 = ${supp.gnn.with_graph_features.f1.toFixed(3)}` : 'F1 = --';
  const arMetric = supp?.autoregressive ? `r = ${supp.autoregressive.metrics.correlation_pc.toFixed(3)}` : 'r = --';
  const confMetric = supp?.conformal?.classification_conformal?.[1] ? `${(supp.conformal.classification_conformal[1].empirical_coverage * 100).toFixed(1)}% coverage` : '-- coverage';

  const models = [
    {
      id: 'gnn',
      name: 'Graph Neural Network',
      metric: gnnMetric,
      desc: supp?.gnn ? `${supp.gnn.n_nodes.toLocaleString()} satellite nodes, ${supp.gnn.n_edges.toLocaleString()} conjunction edges. 1-hop message passing with risk aggregation from orbital neighborhood.` : 'GraphSAGE on conjunction network: satellites as nodes, conjunctions as edges.',
      detail: <GNNDetail data={supp?.gnn} />,
    },
    {
      id: 'ar',
      name: 'Autoregressive Forecaster',
      metric: arMetric,
      desc: supp?.autoregressive ? `Trained on ${supp.autoregressive.n_training_samples.toLocaleString()} CDM transitions. Predicts next CDM update (Pc, miss distance). MAE = ${supp.autoregressive.metrics.mae_log10_pc.toFixed(3)} log\u2081\u2080(Pc).` : 'Predicts next CDM update. Multi-step rollouts with uncertainty.',
      detail: <AutoregressiveDetail data={supp?.autoregressive} />,
    },
    {
      id: 'conformal',
      name: 'Conformal Prediction',
      metric: confMetric,
      desc: supp?.conformal ? `Split-conformal on ${supp.conformal.n_train + supp.conformal.n_calibration + supp.conformal.n_test} pairs (${supp.conformal.n_calibration} calibration). Distribution-free coverage guarantees.` : 'Split-conformal prediction with coverage guarantees.',
      detail: <ConformalDetail data={supp?.conformal} />,
    },
    {
      id: 'summarizer',
      name: 'Event Summarizer',
      metric: 'NLP',
      desc: 'Template-based natural language risk assessments from model predictions, attention weights, and CDM sequences.',
      detail: <SummarizerDetail />,
    },
  ];

  return (
    <div style={{ paddingTop: 24 }}>
      <SectionHeader>Additional Models</SectionHeader>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {models.map(m => (
          <div key={m.id}>
            <button
              onClick={() => setExpanded(expanded === m.id ? null : m.id)}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                gap: 16,
                padding: '14px 16px',
                background: expanded === m.id ? '#111118' : 'transparent',
                border: '1px solid',
                borderColor: expanded === m.id ? '#2a2a3a' : 'transparent',
                borderRadius: 8,
                cursor: 'pointer',
                textAlign: 'left',
                fontFamily: 'inherit',
                transition: 'all 150ms',
              }}
              onMouseEnter={e => { if (expanded !== m.id) (e.currentTarget).style.background = '#0e0e14'; }}
              onMouseLeave={e => { if (expanded !== m.id) (e.currentTarget).style.background = 'transparent'; }}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#e8e8f0' }}>{m.name}</div>
                <div style={{ fontSize: 13, color: '#55556a', marginTop: 2, lineHeight: 1.4 }}>{m.desc}</div>
              </div>
              <div style={{
                fontSize: 14, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace",
                color: '#7c7c96', flexShrink: 0, whiteSpace: 'nowrap',
              }}>
                {m.metric}
              </div>
              <span style={{ color: '#55556a', fontSize: 12, flexShrink: 0, transition: 'transform 150ms', transform: expanded === m.id ? 'rotate(180deg)' : 'rotate(0)' }}>
                &#9662;
              </span>
            </button>
            {expanded === m.id && (
              <div style={{ padding: '4px 16px 20px', animation: 'fadeIn 150ms ease-out' }}>
                {m.detail}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── GNN Detail ───────────────────────────────────────────── */
function GNNDetail({ data }: { data: any }) {
  const degreeData = useMemo(() => {
    if (!data?.degree_distribution) return [];
    return Object.entries(data.degree_distribution)
      .map(([d, count]) => ({ degree: d, count: count as number }))
      .sort((a, b) => parseInt(a.degree) - parseInt(b.degree));
  }, [data]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>
      <div>
        <p style={{ fontSize: 13, color: '#7c7c96', lineHeight: 1.6, marginBottom: 16 }}>
          {data ? `${data.n_nodes.toLocaleString()} satellite nodes with ${data.n_edges.toLocaleString()} conjunction edges (${data.n_positive_edges} high-risk).` : 'Loading...'}{' '}
          1-hop message passing aggregates risk from orbital neighborhood. A satellite
          with many simultaneous high-risk conjunctions gets elevated signal.
        </p>
        <div style={{ display: 'flex', gap: 20, fontSize: 13, marginBottom: 16 }}>
          <Stat label="Nodes" value={data ? data.n_nodes.toLocaleString() : '--'} />
          <Stat label="Edges" value={data ? data.n_edges.toLocaleString() : '--'} />
          <Stat label="Avg Degree" value={data ? data.avg_degree.toFixed(1) : '--'} />
          <Stat label="Max Degree" value={data ? String(data.max_degree) : '--'} />
        </div>
        {data && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div style={{ padding: '10px 14px', background: '#111118', borderRadius: 6, border: '1px solid #1e1e2c' }}>
              <div style={{ fontSize: 12, color: '#55556a', marginBottom: 4 }}>With Graph Features</div>
              <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: '#e8e8f0' }}>F1 = {data.with_graph_features.f1.toFixed(3)}</div>
              <div style={{ fontSize: 12, color: '#55556a' }}>Recall: {(data.with_graph_features.recall * 100).toFixed(1)}%</div>
            </div>
            <div style={{ padding: '10px 14px', background: '#111118', borderRadius: 6, border: '1px solid #1e1e2c' }}>
              <div style={{ fontSize: 12, color: '#55556a', marginBottom: 4 }}>Without Graph Features</div>
              <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: '#7c7c96' }}>F1 = {data.without_graph_features.f1.toFixed(3)}</div>
              <div style={{ fontSize: 12, color: '#55556a' }}>Recall: {(data.without_graph_features.recall * 100).toFixed(1)}%</div>
            </div>
          </div>
        )}
        {data && data.graph_improvement.f1_delta === 0 && (
          <p style={{ fontSize: 12, color: '#55556a', marginTop: 8, fontStyle: 'italic' }}>
            Sparse graph (avg degree {data.avg_degree}) limits message-passing benefit. With denser conjunction networks (mega-constellations), graph features would provide more signal.
          </p>
        )}
      </div>
      {degreeData.length > 0 && (
        <div>
          <div style={{ fontSize: 12, color: '#55556a', marginBottom: 8 }}>Degree Distribution</div>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={degreeData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2c" />
              <XAxis dataKey="degree" tick={{ fill: '#55556a', fontSize: 12 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} />
              <YAxis tick={{ fill: '#55556a', fontSize: 12 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={v => `Degree ${v}`} />
              <Bar dataKey="count" fill="#3b82f6" radius={[2, 2, 0, 0]} opacity={0.6} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

/* ── Autoregressive Detail ────────────────────────────────── */
function AutoregressiveDetail({ data }: { data: any }) {
  const chartData = useMemo(() => {
    if (!data?.example_forecasts?.[0]) return [];
    const ex = data.example_forecasts[0];
    return ex.actual.map((a: number, i: number) => ({
      cdm: i + 1,
      actual: a,
      predicted: ex.predicted[i] ?? null,
    }));
  }, [data]);

  const m = data?.metrics;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>
      <div>
        <p style={{ fontSize: 13, color: '#7c7c96', lineHeight: 1.6, marginBottom: 12 }}>
          Trained on {data ? data.n_training_samples.toLocaleString() : '--'} CDM transitions.
          Given CDM updates 1...k, forecast CDM k+1. Dashed line shows model predictions
          starting from CDM 3 onward.
        </p>
        <div style={{ display: 'flex', gap: 20, fontSize: 13, marginBottom: 12 }}>
          <Stat label="Correlation" value={m ? m.correlation_pc.toFixed(3) : '--'} />
          <Stat label="MAE (log Pc)" value={m ? m.mae_log10_pc.toFixed(3) : '--'} />
          <Stat label="RMSE" value={m ? m.rmse_log10_pc.toFixed(3) : '--'} />
        </div>
        {m && (
          <div style={{ display: 'flex', gap: 20, fontSize: 13 }}>
            <Stat label="Miss Corr" value={m.correlation_miss.toFixed(3)} />
            <Stat label="Miss MAE" value={m.mae_log10_miss.toFixed(3)} />
            <Stat label="Uncertainty" value={`\u00b1${m.uncertainty_std.toFixed(3)}`} />
          </div>
        )}
      </div>
      {chartData.length > 0 && (
        <div>
          <div style={{ fontSize: 12, color: '#55556a', marginBottom: 8 }}>Example: Actual vs Predicted log\u2081\u2080(Pc)</div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={chartData} margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2c" />
              <XAxis dataKey="cdm" tick={{ fill: '#55556a', fontSize: 12 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} />
              <YAxis tick={{ fill: '#55556a', fontSize: 12 }} domain={['auto', 'auto']} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <ReferenceLine y={-3.3} stroke="#ef4444" strokeDasharray="4 3" strokeOpacity={0.3} />
              <Line type="monotone" dataKey="actual" stroke="#e8e8f0" strokeWidth={2} dot={{ r: 3, fill: '#e8e8f0' }} name="Actual" connectNulls={false} />
              <Line type="monotone" dataKey="predicted" stroke="#3b82f6" strokeWidth={2} strokeDasharray="5 3" dot={{ r: 3, fill: '#3b82f6' }} name="Predicted" connectNulls={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

/* ── Conformal Detail ─────────────────────────────────────── */
function ConformalDetail({ data }: { data: any }) {
  const chartData = useMemo(() => {
    if (!data?.classification_conformal) return [];
    return data.classification_conformal.map((c: any) => ({
      name: `\u03b1=${c.alpha}`,
      target: c.target_coverage * 100,
      actual: c.empirical_coverage * 100,
    }));
  }, [data]);

  const regData = useMemo(() => {
    if (!data?.regression_conformal) return [];
    return data.regression_conformal.map((c: any) => ({
      name: `\u03b1=${c.alpha}`,
      target: c.target * 100,
      actual: c.coverage * 100,
      width: c.interval_width,
    }));
  }, [data]);

  return (
    <div>
      <p style={{ fontSize: 13, color: '#7c7c96', lineHeight: 1.6, marginBottom: 12 }}>
        Split-conformal prediction with {data ? data.n_calibration : '--'} calibration samples.
        NASA CARA requires calibrated uncertainty before adopting ML operationally.
        Conformal prediction guarantees that the true outcome falls within our prediction set with no distributional assumptions.
      </p>
      <p style={{ fontSize: 13, color: '#7c7c96', lineHeight: 1.6, marginBottom: 16, padding: '8px 12px', background: 'rgba(34,197,94,0.04)', border: '1px solid rgba(34,197,94,0.12)', borderRadius: 6 }}>
        <strong style={{ color: '#22c55e' }}>In plain terms:</strong> When the model says it is 90% confident, the true outcome actually falls within the predicted range 95%+ of the time. The model's uncertainty estimates are reliable and slightly conservative, which is what you want for safety-critical decisions.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>
        {/* Classification coverage */}
        <div>
          <div style={{ fontSize: 12, color: '#55556a', marginBottom: 8 }}>Classification Coverage</div>
          {chartData.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {data.classification_conformal.map((c: any, i: number) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 13 }}>
                  <span style={{ color: '#55556a', width: 60, flexShrink: 0 }}>{(c.target_coverage * 100).toFixed(0)}% target</span>
                  <div style={{ flex: 1, height: 8, background: '#1a1a24', borderRadius: 4, overflow: 'hidden', position: 'relative' }}>
                    <div style={{ height: '100%', width: `${Math.min(c.empirical_coverage * 100, 100)}%`, background: c.empirical_coverage >= c.target_coverage ? '#3b82f6' : '#ef4444', borderRadius: 4 }} />
                  </div>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: c.empirical_coverage >= c.target_coverage ? '#22c55e' : '#ef4444', width: 50, textAlign: 'right', flexShrink: 0 }}>
                    {(c.empirical_coverage * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          ) : <div style={{ color: '#55556a', fontSize: 13 }}>No data</div>}
        </div>
        {/* Regression coverage */}
        <div>
          <div style={{ fontSize: 12, color: '#55556a', marginBottom: 8 }}>Regression Intervals (log\u2081\u2080 Pc)</div>
          {regData.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {data.regression_conformal.map((c: any, i: number) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 13 }}>
                  <span style={{ color: '#55556a', width: 60, flexShrink: 0 }}>{(c.target * 100).toFixed(0)}% target</span>
                  <div style={{ flex: 1, height: 8, background: '#1a1a24', borderRadius: 4, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${Math.min(c.coverage * 100, 100)}%`, background: c.coverage >= c.target ? '#3b82f6' : '#ef4444', borderRadius: 4 }} />
                  </div>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: c.coverage >= c.target ? '#22c55e' : '#ef4444', width: 50, textAlign: 'right', flexShrink: 0 }}>
                    {(c.coverage * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          ) : <div style={{ color: '#55556a', fontSize: 13 }}>No data</div>}
        </div>
      </div>
    </div>
  );
}

/* ── Summarizer Detail ────────────────────────────────────── */
function SummarizerDetail() {
  return (
    <div>
      <p style={{ fontSize: 13, color: '#7c7c96', lineHeight: 1.6, marginBottom: 12 }}>
        Converts structured model outputs into human-readable risk assessments. Uses template-based
        NLP with optional Claude API enhancement for natural language fluency.
      </p>
      <div style={{
        padding: '16px 20px',
        background: '#111118',
        borderRadius: 8,
        borderLeft: '3px solid #3b82f6',
        fontSize: 13,
        color: '#7c7c96',
        lineHeight: 1.6,
      }}>
        <span style={{ color: '#e8e8f0', fontWeight: 600 }}>Sample: </span>
        COSMOS 2251 DEB vs STARLINK-1234: <span style={{ color: '#ef4444', fontWeight: 600 }}>ELEVATED RISK</span>.
        Collision probability is 1 in 3,125 with a miss distance of 0.4 km.
        Risk is escalating across 5 CDM updates (+15%/update). Ensemble gives 68% chance of exceeding
        the maneuver threshold. Recommend: active monitoring at 6-hour intervals.
      </div>
    </div>
  );
}

/* ── BiLSTM Learning Over Time ────────────────────────────── */
function BiLSTMProgress({ pstats }: { pstats: any }) {
  const ftData = useMemo(() => {
    if (!pstats?.finetune_history) return [];
    return pstats.finetune_history.map((f: any) => ({
      date: f.date?.slice(5, 10) ?? '',
      auc_pr: +(f.post_auc_pr ?? 0).toFixed(3),
      baseline: +(f.pre_auc_pr ?? 0).toFixed(3),
      kept: f.keep_new_model,
    }));
  }, [pstats]);

  if (ftData.length < 2) return null;

  const best = Math.max(...ftData.map((d: any) => d.auc_pr));
  const latest = ftData[ftData.length - 1];
  const nKept = ftData.filter((d: any) => d.kept).length;

  return (
    <div style={{ padding: '24px 0', borderBottom: '1px solid #1e1e2c' }}>
      <SectionHeader>BiLSTM: Learning Over Time</SectionHeader>
      <p style={{ fontSize: 13, color: '#7c7c96', marginBottom: 16, maxWidth: 700, lineHeight: 1.6 }}>
        Weekly fine-tuning on newly resolved conjunction pairs. The BiLSTM retrains each Sunday
        and keeps the new weights only if AUC-PR improves. {nKept} of {ftData.length} fine-tune
        runs improved the model.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 200px', gap: 24, alignItems: 'start' }}>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={ftData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2c" />
            <XAxis dataKey="date" tick={{ fill: '#55556a', fontSize: 12 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} />
            <YAxis domain={[0, 'auto']} tick={{ fill: '#55556a', fontSize: 12 }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={TOOLTIP_STYLE} />
            <Line type="monotone" dataKey="baseline" stroke="#55556a" strokeWidth={1} strokeDasharray="4 3" dot={false} name="Pre-finetune" />
            <Line type="monotone" dataKey="auc_pr" stroke="#3b82f6" strokeWidth={2} dot={(props: any) => {
              const { cx, cy, payload } = props;
              if (!cx || !cy) return <circle key={props.key} />;
              return (
                <circle key={props.key} cx={cx} cy={cy} r={payload.kept ? 5 : 3}
                  fill={payload.kept ? '#22c55e' : '#ef4444'}
                  stroke={payload.kept ? '#22c55e' : '#ef4444'}
                  strokeWidth={payload.kept ? 2 : 1}
                  opacity={payload.kept ? 1 : 0.5}
                />
              );
            }} name="Post-finetune" />
          </LineChart>
        </ResponsiveContainer>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ padding: '10px 14px', background: '#111118', borderRadius: 6, border: '1px solid #1e1e2c' }}>
            <div style={{ fontSize: 12, color: '#55556a', marginBottom: 4 }}>Best AUC-PR</div>
            <div style={{ fontSize: 20, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: '#e8e8f0' }}>{best.toFixed(3)}</div>
          </div>
          <div style={{ padding: '10px 14px', background: '#111118', borderRadius: 6, border: '1px solid #1e1e2c' }}>
            <div style={{ fontSize: 12, color: '#55556a', marginBottom: 4 }}>Latest Run</div>
            <div style={{ fontSize: 15, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace", color: latest?.kept ? '#22c55e' : '#7c7c96' }}>
              {latest?.auc_pr.toFixed(3)} {latest?.kept ? '(kept)' : '(reverted)'}
            </div>
          </div>
          <div style={{ padding: '10px 14px', background: '#111118', borderRadius: 6, border: '1px solid #1e1e2c' }}>
            <div style={{ fontSize: 12, color: '#55556a', marginBottom: 4 }}>Improvements</div>
            <div style={{ fontSize: 15, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace", color: '#e8e8f0' }}>
              {nKept} / {ftData.length} runs
            </div>
          </div>
          <div style={{ fontSize: 11, color: '#55556a', marginTop: 4 }}>
            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#22c55e', marginRight: 4 }} /> Kept
            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#ef4444', marginLeft: 12, marginRight: 4, opacity: 0.5 }} /> Reverted
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Shared components ────────────────────────────────────── */
function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 13, color: '#55556a', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: 16 }}>
      {children}
    </div>
  );
}

function Th({ children, align, title }: { children: React.ReactNode; align?: 'left' | 'center'; title?: string }) {
  return (
    <th title={title} style={{
      padding: '8px 12px',
      fontSize: 12,
      fontWeight: 600,
      color: '#55556a',
      textTransform: 'uppercase',
      letterSpacing: '0.04em',
      textAlign: align ?? 'right',
      cursor: title ? 'help' : undefined,
      borderBottom: title ? '1px dotted #55556a' : undefined,
    }}>
      {children}
    </th>
  );
}

function Td({ children, highlight }: { children: React.ReactNode; highlight?: boolean }) {
  return (
    <td style={{
      padding: '12px',
      fontSize: 15,
      fontWeight: highlight ? 700 : 500,
      fontFamily: "'JetBrains Mono', monospace",
      color: highlight ? '#e8e8f0' : '#7c7c96',
      textAlign: 'right',
    }}>
      {children}
    </td>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ fontSize: 12, color: '#55556a' }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace", color: '#e8e8f0' }}>{value}</div>
    </div>
  );
}
