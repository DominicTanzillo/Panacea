import { useState, useEffect, useCallback, useRef } from 'react';
import type { TLERecord, SatellitePosition, ObjectGroup } from '../lib/types';
import { OBJECT_GROUPS } from '../lib/types';
import { propagateSatellite } from '../lib/orbital';

interface UseSatellitesResult {
  satellites: SatellitePosition[];
  loading: boolean;
  error: string | null;
  loadedGroups: string[];
  totalTLEs: number;
  lastUpdate: Date | null;
  toggleGroup: (groupId: string) => void;
  groups: ObjectGroup[];
  refresh: () => void;
}

// Priority order: specific groups win over generic "active"
// e.g. Starlink (constellation) should be blue, not green from active
const TYPE_PRIORITY: Record<string, number> = {
  station: 0,
  constellation: 1,
  debris: 2,
  rocket_body: 3,
  active: 4,
};

// Deduplicate TLEs across groups by NORAD_CAT_ID (highest-priority group wins)
function deduplicateTLEs(
  enabledGroups: ObjectGroup[],
  tleCache: Map<string, TLERecord[]>
): { tle: TLERecord; group: ObjectGroup }[] {
  // Sort by priority: specific groups first, generic "active" last
  const sorted = [...enabledGroups].sort(
    (a, b) => (TYPE_PRIORITY[a.type] ?? 99) - (TYPE_PRIORITY[b.type] ?? 99)
  );

  const seen = new Set<number>();
  const result: { tle: TLERecord; group: ObjectGroup }[] = [];

  for (const group of sorted) {
    const tles = tleCache.get(group.id);
    if (!tles) continue;

    for (const tle of tles) {
      if (!seen.has(tle.NORAD_CAT_ID)) {
        seen.add(tle.NORAD_CAT_ID);
        result.push({ tle, group });
      }
    }
  }

  return result;
}

// Propagate a batch of TLEs, yielding to the main thread between chunks
function propagateBatch(
  items: { tle: TLERecord; group: ObjectGroup }[],
  date: Date
): SatellitePosition[] {
  const positions: SatellitePosition[] = [];
  for (const { tle, group } of items) {
    const pos = propagateSatellite(tle, date, group.id, group.type);
    if (pos) positions.push(pos);
  }
  return positions;
}

export function useSatellites(): UseSatellitesResult {
  const [tleCache, setTleCache] = useState<Map<string, TLERecord[]>>(new Map());
  const [groups, setGroups] = useState<ObjectGroup[]>(OBJECT_GROUPS);
  const [satellites, setSatellites] = useState<SatellitePosition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const propagatingRef = useRef(false);

  // Fetch TLE data for enabled groups
  const fetchGroups = useCallback(async () => {
    setLoading(true);
    setError(null);

    const enabledGroups = groups.filter(g => g.enabled);
    const newCache = new Map(tleCache);
    const promises = enabledGroups
      .filter(g => !newCache.has(g.id))
      .map(async (group) => {
        try {
          const resp = await fetch(group.url);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
          const data: TLERecord[] = await resp.json();
          return { id: group.id, data };
        } catch (err) {
          console.error(`Failed to fetch ${group.id}:`, err);
          return { id: group.id, data: [] as TLERecord[] };
        }
      });

    const results = await Promise.all(promises);
    for (const { id, data } of results) {
      newCache.set(id, data);
    }
    setTleCache(newCache);
    setLoading(false);
    setLastUpdate(new Date());
  }, [groups, tleCache]);

  // Initial fetch
  useEffect(() => {
    fetchGroups();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Chunked propagation that yields to main thread
  const propagateAll = useCallback(async () => {
    if (propagatingRef.current) return;
    propagatingRef.current = true;

    const enabledGroups = groups.filter(g => g.enabled);
    const items = deduplicateTLEs(enabledGroups, tleCache);
    const now = new Date();
    const CHUNK_SIZE = 500;
    const allPositions: SatellitePosition[] = [];

    for (let i = 0; i < items.length; i += CHUNK_SIZE) {
      const chunk = items.slice(i, i + CHUNK_SIZE);
      const positions = propagateBatch(chunk, now);
      allPositions.push(...positions);

      // Yield to main thread between chunks so the UI stays responsive
      if (i + CHUNK_SIZE < items.length) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    setSatellites(allPositions);
    propagatingRef.current = false;
  }, [tleCache, groups]);

  // Propagate when data or group selection changes
  useEffect(() => {
    propagateAll();
  }, [propagateAll]);

  // Continuous propagation loop — adaptive interval based on object count
  useEffect(() => {
    const enabledGroups = groups.filter(g => g.enabled);
    let totalCount = 0;
    for (const group of enabledGroups) {
      totalCount += tleCache.get(group.id)?.length ?? 0;
    }

    // Scale interval: 2s for <2k objects, up to 10s for 20k+
    const interval = Math.min(10000, Math.max(2000, totalCount * 0.5));

    const timer = setInterval(() => {
      propagateAll();
    }, interval);
    return () => clearInterval(timer);
  }, [tleCache, groups, propagateAll]);

  const toggleGroup = useCallback((groupId: string) => {
    setGroups(prev => prev.map(g =>
      g.id === groupId ? { ...g, enabled: !g.enabled } : g
    ));
  }, []);

  const loadedGroups = Array.from(tleCache.keys());
  const totalTLEs = Array.from(tleCache.values()).reduce((sum, arr) => sum + arr.length, 0);

  return {
    satellites,
    loading,
    error,
    loadedGroups,
    totalTLEs,
    lastUpdate,
    toggleGroup,
    groups,
    refresh: fetchGroups,
  };
}
