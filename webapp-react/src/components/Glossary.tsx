import { useState } from 'react';

/* ── domain‑term definitions ────────────────────────────────────── */
export const GLOSSARY: Record<string, { short: string; long: string }> = {
  CDM: {
    short: 'Conjunction Data Message — official warning about a potential satellite collision',
    long: 'A Conjunction Data Message (CDM) is issued by the 18th Space Defense Squadron via Space-Track.org when two objects in orbit are predicted to pass close to each other. Each CDM carries collision probability (Pc), miss distance, and timing data. Multiple CDMs are issued as an event evolves.',
  },
  Pc: {
    short: 'Collision Probability — likelihood of impact between two objects (0 to 1)',
    long: 'Pc (probability of collision) quantifies how likely two objects are to collide during a close approach. Expressed as a decimal: Pc = 1e-4 means a 1-in-10,000 chance. Operators begin planning avoidance maneuvers when Pc exceeds 5e-4 (1 in 2,000).',
  },
  TCA: {
    short: 'Time of Closest Approach — moment when two objects pass nearest to each other',
    long: 'The Time of Closest Approach (TCA) is the predicted instant when two orbiting objects will be at their minimum separation distance. All collision predictions are relative to TCA. Once TCA passes without collision, the event is resolved.',
  },
  'Miss Distance': {
    short: 'Predicted closest approach distance between two objects, in kilometers',
    long: 'Miss distance is the minimum predicted separation between two objects during a close approach, measured in kilometers. Smaller miss distances indicate higher risk. CDM miss distances under 1 km warrant serious attention.',
  },
  SGP4: {
    short: 'Simplified General Perturbations — algorithm for predicting satellite positions',
    long: 'SGP4 is a mathematical model for forecasting satellite positions using published orbital elements (TLEs). It accounts for Earth\'s gravity, atmospheric drag, and solar/lunar perturbations. SGP4 is less accurate than official CDM predictions but works for all cataloged objects.',
  },
  TLE: {
    short: 'Two-Line Element Set — standardized format for satellite orbital parameters',
    long: 'A TLE (Two-Line Element Set) describes a satellite\'s orbit in a standardized format published by CelesTrak and the US Space Force. TLEs are updated regularly and used with SGP4 to compute satellite positions. They are less precise than CDM data.',
  },
  RCS: {
    short: 'Radar Cross-Section — measure of object size/reflectivity',
    long: 'RCS (Radar Cross-Section) indicates an object\'s apparent size to radar. Classified as SMALL (<0.1 m2), MEDIUM (0.1-1.0 m2), or LARGE (>1.0 m2). Larger objects have higher collision probability due to their greater cross-sectional area.',
  },
  'Maneuver Threshold': {
    short: 'Pc level (5e-4) where operators begin planning collision avoidance',
    long: 'The maneuver threshold (Pc = 5e-4, or 1 in 2,000) is the collision probability level at which satellite operators typically begin planning avoidance maneuvers. Below this, most conjunctions resolve without action. Above this, operators commit analyst time and potentially fuel.',
  },
  'Kessler Syndrome': {
    short: 'Cascading chain reaction of orbital collisions creating runaway debris',
    long: 'Kessler Syndrome is a theoretical scenario where collision-generated debris triggers further collisions in a runaway chain reaction, eventually rendering entire orbital shells unusable. First described by NASA scientist Donald Kessler in 1978. Preventing this is the motivation behind collision avoidance systems like PANACEA.',
  },
  Ensemble: {
    short: 'Combination of multiple ML models for more reliable predictions',
    long: 'The PANACEA ensemble combines three models: 40% logistic regression (interpretable, high recall), 30% BiLSTM (temporal patterns), and 30% regression signal (continuous risk). Ensembles typically outperform any single model because different models make different errors.',
  },
  Recall: {
    short: 'Percentage of dangerous events the model successfully catches',
    long: 'Recall measures what fraction of truly dangerous conjunction events the model correctly identifies. 98.7% recall means the model catches ~74 out of 75 escalating events. In safety-critical systems, high recall is prioritized over precision.',
  },
  Precision: {
    short: 'Percentage of model alerts that turn out to be real threats',
    long: 'Precision measures what fraction of the model\'s "action recommended" predictions are genuine escalations. 74.3% precision means ~26% of alerts are false alarms. Lower precision means more analyst workload but fewer missed events.',
  },
  F1: {
    short: 'Harmonic mean of Precision and Recall — overall classification quality',
    long: 'F1 score balances precision and recall into a single metric. An F1 of 0.847 means the model achieves a strong balance between catching dangerous events (recall) and avoiding false alarms (precision). F1 = 1.0 would be perfect.',
  },
  'Space Weather': {
    short: 'Solar and geomagnetic activity affecting satellite orbits',
    long: 'Space weather includes solar flux (F10.7), geomagnetic storms (Kp index, 0-9 scale), and their amplitude (Ap). High space weather increases atmospheric drag on LEO satellites, making orbit predictions less accurate and potentially affecting collision probability estimates.',
  },
};

/* ── Tooltip component ──────────────────────────────────────────── */
interface TooltipProps {
  term: string;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}

export function GlossaryTooltip({ term, children, style }: TooltipProps) {
  const [show, setShow] = useState(false);
  const entry = GLOSSARY[term];
  if (!entry) return <>{children ?? term}</>;

  return (
    <span
      style={{ position: 'relative', display: 'inline', ...style }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <span
        style={{
          borderBottom: '1px dotted #55556a',
          cursor: 'help',
        }}
        tabIndex={0}
        role="button"
        aria-label={`Definition: ${entry.short}`}
        onFocus={() => setShow(true)}
        onBlur={() => setShow(false)}
      >
        {children ?? term}
      </span>
      {show && (
        <span
          role="tooltip"
          style={{
            position: 'absolute',
            bottom: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            marginBottom: 8,
            background: '#1a1a2e',
            border: '1px solid #333355',
            borderRadius: 8,
            padding: '10px 14px',
            width: 300,
            maxWidth: '90vw',
            color: '#ccc',
            fontSize: 13,
            lineHeight: 1.5,
            zIndex: 9999,
            pointerEvents: 'none',
            boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
          }}
        >
          <strong style={{ color: '#7c8aff', display: 'block', marginBottom: 4 }}>
            {term}
          </strong>
          {entry.short}
        </span>
      )}
    </span>
  );
}

/* ── Full Glossary panel (modal / page) ─────────────────────────── */
interface GlossaryPanelProps {
  onClose: () => void;
}

export function GlossaryPanel({ onClose }: GlossaryPanelProps) {
  const terms = Object.entries(GLOSSARY).sort(([a], [b]) => a.localeCompare(b));

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.7)',
        zIndex: 10000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: '#111118',
          border: '1px solid #333355',
          borderRadius: 12,
          width: '90%',
          maxWidth: 700,
          maxHeight: '80vh',
          overflow: 'auto',
          padding: '28px 32px',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h2 style={{ margin: 0, color: '#e0e0e0', fontSize: 22 }}>Glossary</h2>
          <button
            onClick={onClose}
            aria-label="Close glossary"
            style={{
              background: 'none',
              border: '1px solid #444',
              borderRadius: 6,
              color: '#999',
              cursor: 'pointer',
              fontSize: 18,
              padding: '4px 10px',
            }}
          >
            Close
          </button>
        </div>
        <p style={{ color: '#888', fontSize: 14, marginBottom: 20 }}>
          Key terms used in satellite conjunction assessment and the PANACEA system.
        </p>
        {terms.map(([term, entry]) => (
          <div key={term} style={{ marginBottom: 16, borderBottom: '1px solid #222', paddingBottom: 12 }}>
            <strong style={{ color: '#7c8aff', fontSize: 15 }}>{term}</strong>
            <p style={{ color: '#bbb', fontSize: 13, margin: '4px 0 0' }}>{entry.long}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
