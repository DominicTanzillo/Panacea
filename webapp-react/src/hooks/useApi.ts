import { useState, useEffect, useCallback } from 'react';
import type {
  ModelComparisonResult,
  StalenessResults,
  ScreeningPair,
  StaticAlerts,
} from '../lib/api';
import {
  health as apiHealth,
  getModelComparison,
  getExperimentResults,
  bulkScreen,
} from '../lib/api';
import type { TLERecord } from '../lib/types';

interface UseApiResult {
  healthy: boolean;
  modelComparison: ModelComparisonResult[] | null;
  experimentResults: StalenessResults | null;
  screeningPairs: ScreeningPair[];
  loading: boolean;
  refresh: () => void;
}

export function useApi(tles?: TLERecord[]): UseApiResult {
  const [healthy, setHealthy] = useState(false);
  const [modelComparison, setModelComparison] = useState<ModelComparisonResult[] | null>(null);
  const [experimentResults, setExperimentResults] = useState<StalenessResults | null>(null);
  const [screeningPairs, setScreeningPairs] = useState<ScreeningPair[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    setLoading(true);

    // Check health first
    const healthResp = await apiHealth();
    const isHealthy = healthResp?.status === 'healthy';
    setHealthy(isHealthy);

    if (isHealthy) {
      // Fetch from backend API
      const [comparison, experiment] = await Promise.all([
        getModelComparison(),
        getExperimentResults(),
      ]);

      if (comparison) setModelComparison(comparison);
      if (experiment && !('error' in experiment)) {
        setExperimentResults(experiment);
      }
    } else {
      // Fallback: load static result files bundled in public/
      try {
        const [compResp, expResp] = await Promise.all([
          fetch('./model_comparison.json'),
          fetch('./staleness_experiment.json'),
        ]);
        if (compResp.ok) setModelComparison(await compResp.json());
        if (expResp.ok) setExperimentResults(await expResp.json());
      } catch {
        // Static files not available either — dashboard shows empty state
      }
    }

    setLoading(false);
  }, []);

  // Deferred alerts loading — parse 763KB JSON after globe is interactive
  // to avoid blocking the main thread during initial satellite rendering.
  useEffect(() => {
    const timer = setTimeout(async () => {
      try {
        const alertsResp = await fetch('./latest_alerts.json');
        if (alertsResp.ok) {
          const data: StaticAlerts = await alertsResp.json();
          if (data?.pairs) setScreeningPairs(data.pairs);
        }
      } catch {
        // Static alerts not available
      }
    }, 2000); // 2s delay lets globe render first
    return () => clearTimeout(timer);
  }, []);

  // Initial fetch + poll health every 30s (refetch data if backend comes online)
  useEffect(() => {
    fetchData();
    let wasHealthy = false;
    const timer = setInterval(async () => {
      const resp = await apiHealth();
      const isNowHealthy = resp?.status === 'healthy';
      setHealthy(isNowHealthy);
      if (isNowHealthy && !wasHealthy) fetchData();
      wasHealthy = isNowHealthy;
    }, 30000);
    return () => clearInterval(timer);
  }, [fetchData]);

  // Screen TLEs via backend when available
  useEffect(() => {
    if (!healthy || !tles || tles.length < 100) return;

    // Sample up to 2000 TLEs for screening (avoid overloading)
    const sample = tles.length > 2000
      ? tles.filter((_, i) => i % Math.ceil(tles.length / 2000) === 0)
      : tles;

    bulkScreen(sample as unknown as Record<string, unknown>[], 20).then(resp => {
      if (resp?.pairs) setScreeningPairs(resp.pairs);
    });
  }, [healthy, tles]);

  return {
    healthy,
    modelComparison,
    experimentResults,
    screeningPairs,
    loading,
    refresh: fetchData,
  };
}
