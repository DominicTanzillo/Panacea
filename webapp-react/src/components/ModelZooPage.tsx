import { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LineChart, Line, ReferenceLine,
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

  useEffect(() => {
    if (!visible) return;
    fetch('./cv_results.json')
      .then(r => r.ok ? r.json() : null)
      .then(setCv)
      .catch(() => setCv(null));
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

          {/* ── Section 1: Task & Key Insight ────────────────── */}
          <div style={{ padding: '24px 0 20px', borderBottom: '1px solid #1e1e2c' }}>
            <div style={{ fontSize: 13, color: '#55556a', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: 8 }}>
              Prediction Task
            </div>
            <p style={{ fontSize: 15, color: '#e8e8f0', lineHeight: 1.5, maxWidth: 700, marginBottom: 16 }}>
              Given a sequence of CDM updates for a conjunction pair, predict whether collision probability (Pc)
              will exceed <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#3b82f6' }}>5 &times; 10<sup>-4</sup></span> before
              time of closest approach — the threshold at which operators begin planning avoidance maneuvers.
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
                means an undetected escalation — an operator doesn't plan a maneuver, and a $500M
                satellite is at risk. A false positive means 30 minutes of extra monitoring. We optimize
                for catching every dangerous event, even at the cost of more false alarms.
              </p>
            </div>
          </div>

          {/* ── Section 2: Model Comparison Table ────────────── */}
          <ModelComparisonTable cv={cv} />

          {/* ── Section 3: Per-Fold Stability ────────────────── */}
          <FoldStabilityChart cv={cv} />

          {/* ── Section 4: Ensemble Deep-Dive ────────────────── */}
          <EnsembleSection cv={cv} />

          {/* ── Section 5: Supplementary Models ──────────────── */}
          <SupplementaryModels />

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
        role: 'Production baseline \u00b7 20 features, retrained daily',
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
              <Th>Recall</Th>
              <Th>Missed Events</Th>
              <Th>F1</Th>
              <Th>Precision</Th>
              <Th>Accuracy</Th>
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
        In collision avoidance, a false negative means a missed escalation event — an undetected risk
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
            <WeightBar label="Logistic Regression" weight={40} desc="20 engineered features, retrained daily" />
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
function SupplementaryModels() {
  const [expanded, setExpanded] = useState<string | null>(null);

  const models = [
    {
      id: 'gnn',
      name: 'Graph Neural Network',
      metric: 'F1 = 0.783',
      desc: 'GraphSAGE on conjunction network — satellites as nodes, conjunctions as edges. Captures orbital neighborhood risk that per-pair models miss.',
      detail: <GNNDetail />,
    },
    {
      id: 'ar',
      name: 'Autoregressive Forecaster',
      metric: 'r = 0.931',
      desc: 'BiLSTM predicts next CDM update (Pc, miss distance, time-to-TCA). Multi-step rollouts with MC dropout uncertainty.',
      detail: <AutoregressiveDetail />,
    },
    {
      id: 'conformal',
      name: 'Conformal Prediction',
      metric: '90.8% coverage',
      desc: 'Split-conformal prediction with distribution-free coverage guarantees. Addresses NASA CARA\'s requirement for calibrated uncertainty.',
      detail: <ConformalDetail />,
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
function GNNDetail() {
  const nodes = [
    { x: 200, y: 50, label: 'COSMOS 2251', type: 'debris' },
    { x: 80, y: 130, label: 'STARLINK-1234', type: 'payload' },
    { x: 320, y: 130, label: 'FENGYUN 1C', type: 'debris' },
    { x: 140, y: 220, label: 'CZ-6A R/B', type: 'rb' },
    { x: 260, y: 220, label: 'STARLINK-5678', type: 'payload' },
    { x: 200, y: 300, label: 'IRIDIUM 33', type: 'debris' },
  ];
  const edges = [
    { from: 0, to: 1, risk: 0.82 }, { from: 0, to: 2, risk: 0.91 },
    { from: 1, to: 3, risk: 0.35 }, { from: 2, to: 4, risk: 0.67 },
    { from: 3, to: 5, risk: 0.44 }, { from: 4, to: 5, risk: 0.78 },
  ];
  const typeColor: Record<string, string> = { payload: '#22c55e', debris: '#ef4444', rb: '#f59e0b' };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 24, alignItems: 'start' }}>
      <div>
        <p style={{ fontSize: 13, color: '#7c7c96', lineHeight: 1.6, marginBottom: 12 }}>
          1,212 satellite nodes with conjunction pairs as edges. GraphSAGE message-passing
          aggregates risk from orbital neighborhood — a satellite with many simultaneous
          high-risk conjunctions gets elevated signal that per-pair models cannot capture.
        </p>
        <div style={{ display: 'flex', gap: 16, fontSize: 13 }}>
          <Stat label="Nodes" value="1,212" />
          <Stat label="Edges" value="3,847" />
          <Stat label="F1" value="0.783" />
        </div>
      </div>
      <svg viewBox="0 0 400 340" style={{ width: '100%' }}>
        {edges.map((e, i) => {
          const a = nodes[e.from], b = nodes[e.to];
          return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={e.risk > 0.7 ? '#ef4444' : e.risk > 0.4 ? '#f59e0b' : '#2a2a3a'} strokeWidth={e.risk > 0.7 ? 2 : 1.5} strokeOpacity={0.5} />;
        })}
        {nodes.map((n, i) => (
          <g key={i}>
            <circle cx={n.x} cy={n.y} r={16} fill={typeColor[n.type] + '15'} stroke={typeColor[n.type]} strokeWidth={1.2} />
            <text x={n.x} y={n.y + 4} textAnchor="middle" fill="#7c7c96" fontSize={8} fontWeight="500">{n.label.split(' ')[0]}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

/* ── Autoregressive Detail ────────────────────────────────── */
function AutoregressiveDetail() {
  const data = [
    { cdm: 1, actual: -3.8, predicted: -3.9 }, { cdm: 2, actual: -3.6, predicted: -3.7 },
    { cdm: 3, actual: -3.4, predicted: -3.5 }, { cdm: 4, actual: -3.2, predicted: -3.3 },
    { cdm: 5, actual: -3.1, predicted: -3.2 }, { cdm: 6, actual: null, predicted: -3.0 },
    { cdm: 7, actual: null, predicted: -2.9 }, { cdm: 8, actual: null, predicted: -2.85 },
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>
      <div>
        <p style={{ fontSize: 13, color: '#7c7c96', lineHeight: 1.6, marginBottom: 12 }}>
          Given CDM updates 1...k, forecast CDM k+1. The dashed line shows predicted
          future trajectory beyond observed data. Monte Carlo dropout provides uncertainty
          estimates for each forecast step.
        </p>
        <div style={{ display: 'flex', gap: 16, fontSize: 13 }}>
          <Stat label="Correlation" value="0.931" />
          <Stat label="MAE" value="0.094" />
        </div>
      </div>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2c" />
          <XAxis dataKey="cdm" tick={{ fill: '#55556a', fontSize: 12 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} />
          <YAxis tick={{ fill: '#55556a', fontSize: 12 }} domain={[-4, -2.5]} axisLine={false} tickLine={false} />
          <Tooltip contentStyle={TOOLTIP_STYLE} />
          <ReferenceLine y={-3.3} stroke="#ef4444" strokeDasharray="4 3" strokeOpacity={0.5} />
          <Line type="monotone" dataKey="actual" stroke="#e8e8f0" strokeWidth={2} dot={{ r: 3, fill: '#e8e8f0' }} name="Actual" connectNulls={false} />
          <Line type="monotone" dataKey="predicted" stroke="#3b82f6" strokeWidth={2} strokeDasharray="5 3" dot={{ r: 3, fill: '#3b82f6' }} name="Predicted" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Conformal Detail ─────────────────────────────────────── */
function ConformalDetail() {
  const data = [
    { name: 'Target', coverage: 90 },
    { name: 'Regression', coverage: 90.8 },
    { name: 'Classification', coverage: 91.7 },
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>
      <div>
        <p style={{ fontSize: 13, color: '#7c7c96', lineHeight: 1.6, marginBottom: 12 }}>
          NASA CARA requires calibrated uncertainty before adopting ML operationally.
          Conformal prediction guarantees that the true Pc falls within our prediction
          interval at least 90% of the time — with no distributional assumptions.
        </p>
        <div style={{ display: 'flex', gap: 16, fontSize: 13 }}>
          <Stat label="Coverage" value="90.8%" />
          <Stat label="Target" value="90.0%" />
        </div>
      </div>
      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 80, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2c" />
          <XAxis type="number" domain={[85, 95]} tick={{ fill: '#55556a', fontSize: 12 }} tickFormatter={v => `${v}%`} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="name" tick={{ fill: '#7c7c96', fontSize: 12 }} axisLine={false} tickLine={false} />
          <Tooltip contentStyle={TOOLTIP_STYLE} />
          <Bar dataKey="coverage" radius={[0, 3, 3, 0]}>
            <Cell fill="#55556a" />
            <Cell fill="#3b82f6" />
            <Cell fill="#3b82f6" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
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

/* ── Shared components ────────────────────────────────────── */
function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 13, color: '#55556a', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: 16 }}>
      {children}
    </div>
  );
}

function Th({ children, align }: { children: React.ReactNode; align?: 'left' | 'center' }) {
  return (
    <th style={{
      padding: '8px 12px',
      fontSize: 12,
      fontWeight: 600,
      color: '#55556a',
      textTransform: 'uppercase',
      letterSpacing: '0.04em',
      textAlign: align ?? 'right',
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
