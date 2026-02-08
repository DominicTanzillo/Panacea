import { useState, Suspense } from 'react';
import { Globe } from './components/Globe';
import { Header } from './components/Header';
import { InfoPanel } from './components/InfoPanel';
import { StatusBar } from './components/StatusBar';
import { useSatellites } from './hooks/useSatellites';
import type { SatellitePosition } from './lib/types';

function LoadingScreen() {
  return (
    <div className="w-full h-full flex flex-col items-center justify-center bg-[var(--color-bg)]">
      <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[var(--color-accent)] to-[#8b5cf6] flex items-center justify-center text-xl font-bold mb-4 animate-pulse">
        P
      </div>
      <p className="text-sm text-[var(--color-text-muted)]">
        Loading orbital data from CelesTrak...
      </p>
      <div className="mt-3 w-48 h-1 rounded-full bg-[var(--color-surface-2)] overflow-hidden">
        <div className="h-full bg-[var(--color-accent)] rounded-full animate-pulse" style={{ width: '60%' }} />
      </div>
    </div>
  );
}

function App() {
  const {
    satellites,
    loading,
    totalTLEs,
    groups,
    toggleGroup,
    lastUpdate,
  } = useSatellites();

  const [selectedSatellite, setSelectedSatellite] = useState<SatellitePosition | null>(null);

  return (
    <div className="w-full h-full relative">
      <Header />

      <Suspense fallback={<LoadingScreen />}>
        <Globe
          satellites={satellites}
          onSelectSatellite={setSelectedSatellite}
          selectedSatellite={selectedSatellite}
        />
      </Suspense>

      <InfoPanel
        satellite={selectedSatellite}
        onClose={() => setSelectedSatellite(null)}
      />

      <StatusBar
        satellites={satellites}
        loading={loading}
        totalTLEs={totalTLEs}
        groups={groups}
        onToggleGroup={toggleGroup}
        lastUpdate={lastUpdate}
      />
    </div>
  );
}

export default App;
