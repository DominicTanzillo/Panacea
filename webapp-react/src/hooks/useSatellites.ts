import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import type { TLERecord, SatellitePosition, ObjectGroup } from '../lib/types';
import { OBJECT_GROUPS } from '../lib/types';
import { propagateSatellite } from '../lib/orbital';

interface UseSatellitesResult {
  satellites: SatellitePosition[];
  allTLEs: TLERecord[];
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
  const initialLoadDoneRef = useRef(false);

  // Fetch TLE data: static fallback FIRST (guaranteed same-origin), then
  // try CelesTrak as an upgrade for fresher data. This ensures satellites
  // always render even when CelesTrak is CORS-blocked.
  const fetchGroups = useCallback(async () => {
    setLoading(true);
    setError(null);

    const enabledGroups = groups.filter(g => g.enabled);
    const newCache = new Map(tleCache);
    const toFetch = enabledGroups.filter(g => !newCache.has(g.id));

    if (toFetch.length === 0) {
      setLoading(false);
      return;
    }

    // Phase 1: Load static fallback (same-origin, always works)
    try {
      const resp = await fetch('./latest_tles.json');
      if (resp.ok) {
        const allTles: (TLERecord & { _group?: string })[] = await resp.json();
        // Distribute TLEs into groups by _group tag
        for (const group of toFetch) {
          const groupTles = allTles.filter(t => t._group === group.id);
          if (groupTles.length > 0) {
            newCache.set(group.id, groupTles);
          }
        }
        // Any TLEs without a matching group go into 'active'
        const assigned = new Set(toFetch.map(g => g.id));
        const unmatched = allTles.filter(t => !t._group || !assigned.has(t._group));
        if (unmatched.length > 0) {
          const existing = newCache.get('active') || [];
          const existingIds = new Set(existing.map(t => t.NORAD_CAT_ID));
          const merged = [...existing, ...unmatched.filter(t => !existingIds.has(t.NORAD_CAT_ID))];
          newCache.set('active', merged);
        }
        console.log(`Static TLEs loaded: ${allTles.length} satellites`);
      }
    } catch (fallbackErr) {
      console.error('Static TLE load failed:', fallbackErr);
    }

    // Phase 2: Try CelesTrak for fresher data (may be CORS-blocked)
    const promises = toFetch.map(async (group) => {
      try {
        const resp = await fetch(group.url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data: TLERecord[] = await resp.json();
        if (data.length > 0) return { id: group.id, data };
      } catch {
        // CelesTrak blocked — static fallback already loaded, no problem
      }
      return null;
    });

    const results = await Promise.all(promises);
    let upgraded = 0;
    for (const r of results) {
      if (r && r.data.length > 0) {
        newCache.set(r.id, r.data);
        upgraded++;
      }
    }
    if (upgraded > 0) {
      console.log(`CelesTrak upgraded ${upgraded} groups with fresh TLEs`);
    } else {
      console.log('CelesTrak unavailable — using static TLEs (updated daily by pipeline)');
    }

    if (newCache.size === 0) {
      setError('Unable to load satellite data');
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

    // Progressive rendering: stream satellites onto globe during initial load
    const progressive = !initialLoadDoneRef.current && items.length > 0;
    const PROGRESSIVE_BATCH = 2000;
    let nextUpdate = PROGRESSIVE_BATCH;

    for (let i = 0; i < items.length; i += CHUNK_SIZE) {
      const chunk = items.slice(i, i + CHUNK_SIZE);
      const positions = propagateBatch(chunk, now);
      allPositions.push(...positions);

      // Push partial results so satellites appear incrementally
      if (progressive && allPositions.length >= nextUpdate) {
        setSatellites([...allPositions]);
        nextUpdate += PROGRESSIVE_BATCH;
      }

      // Yield to main thread between chunks so the UI stays responsive
      if (i + CHUNK_SIZE < items.length) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    setSatellites(allPositions);
    if (items.length > 0) initialLoadDoneRef.current = true;
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

  // Expose deduplicated raw TLEs for backend screening
  const allTLEs = useMemo(() => {
    const enabledGroups = groups.filter(g => g.enabled);
    return deduplicateTLEs(enabledGroups, tleCache).map(item => item.tle);
  }, [groups, tleCache]);

  return {
    satellites,
    allTLEs,
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
