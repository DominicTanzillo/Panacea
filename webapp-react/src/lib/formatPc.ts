/**
 * Human-friendly collision probability formatting.
 * Replaces cryptic "3.2e-4" with "1 in 3,125" or "0.032%".
 */

export interface PcDisplay {
  /** Primary human-readable text */
  primary: string;
  /** Secondary (scientific notation for experts) */
  scientific: string;
  /** Risk color */
  color: string;
  /** Severity level */
  level: 'critical' | 'high' | 'moderate' | 'low';
}

export function formatPcHuman(pc: number): PcDisplay {
  if (!pc || pc <= 0) {
    return { primary: '--', scientific: '--', color: '#666', level: 'low' };
  }

  if (pc >= 0.01) {
    return {
      primary: `${(pc * 100).toFixed(1)}%`,
      scientific: pc.toExponential(1),
      color: '#ff4f5a',
      level: 'critical',
    };
  }

  if (pc >= 5e-4) {
    const ratio = Math.round(1 / pc);
    return {
      primary: `1 in ${ratio.toLocaleString()}`,
      scientific: pc.toExponential(1),
      color: '#ff4f5a',
      level: 'high',
    };
  }

  if (pc >= 1e-4) {
    const ratio = Math.round(1 / pc);
    return {
      primary: `1 in ${ratio.toLocaleString()}`,
      scientific: pc.toExponential(1),
      color: '#ffb84f',
      level: 'moderate',
    };
  }

  if (pc >= 1e-7) {
    const ratio = Math.round(1 / pc);
    return {
      primary: `1 in ${ratio.toLocaleString()}`,
      scientific: pc.toExponential(1),
      color: '#4f8aff',
      level: 'low',
    };
  }

  return {
    primary: '< 1 in 10M',
    scientific: pc.toExponential(1),
    color: '#4fff8a',
    level: 'low',
  };
}

/** Short format: "1 in 3K" for tight spaces */
export function formatPcShort(pc: number): string {
  if (!pc || pc <= 0) return '--';
  if (pc >= 0.01) return `${(pc * 100).toFixed(1)}%`;
  if (pc >= 1e-3) return `1 in ${Math.round(1 / pc).toLocaleString()}`;
  const ratio = Math.round(1 / pc);
  if (ratio >= 1_000_000) return `1 in ${(ratio / 1_000_000).toFixed(0)}M`;
  if (ratio >= 1_000) return `1 in ${(ratio / 1_000).toFixed(0)}K`;
  return `1 in ${ratio.toLocaleString()}`;
}

/** Percentage bar width (0-100) on log scale from 1e-7 to 1e-1 */
export function pcToBarWidth(pc: number): number {
  if (!pc || pc <= 0) return 0;
  const logPc = Math.log10(Math.max(pc, 1e-8));
  // Map -7 -> 0%, -1 -> 100%
  return Math.max(0, Math.min(100, ((logPc + 7) / 6) * 100));
}
