import type { SatellitePosition } from '../lib/types';
import { TYPE_COLORS } from '../lib/types';

interface InfoPanelProps {
  satellite: SatellitePosition | null;
  onClose: () => void;
}

export function InfoPanel({ satellite, onClose }: InfoPanelProps) {
  if (!satellite) return null;

  const color = TYPE_COLORS[satellite.type];

  return (
    <div className="absolute right-4 top-20 w-80 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]/95 backdrop-blur-md p-4 shadow-2xl z-20">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
          <h3 className="font-semibold text-sm truncate max-w-[200px]">
            {satellite.name}
          </h3>
        </div>
        <button
          onClick={onClose}
          className="text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors text-lg leading-none"
        >
          x
        </button>
      </div>

      <div className="space-y-2 text-xs">
        <Row label="NORAD ID" value={String(satellite.noradId)} />
        <Row label="COSPAR ID" value={satellite.objectId} />
        <Row label="Type" value={satellite.type.replace('_', ' ')} />
        <Row label="Group" value={satellite.group} />

        <div className="border-t border-[var(--color-border)] my-2" />

        <Row label="Altitude" value={`${satellite.alt.toFixed(1)} km`} />
        <Row label="Latitude" value={`${satellite.lat.toFixed(3)}deg`} />
        <Row label="Longitude" value={`${satellite.lon.toFixed(3)}deg`} />

        <div className="border-t border-[var(--color-border)] my-2" />

        <Row label="ECI X" value={`${satellite.x.toFixed(1)} km`} mono />
        <Row label="ECI Y" value={`${satellite.y.toFixed(1)} km`} mono />
        <Row label="ECI Z" value={`${satellite.z.toFixed(1)} km`} mono />
      </div>

      <div className="mt-3 pt-3 border-t border-[var(--color-border)]">
        <div className="text-[10px] text-[var(--color-text-muted)] text-center">
          Position computed via SGP4 propagation from CelesTrak TLEs
        </div>
      </div>
    </div>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-[var(--color-text-muted)]">{label}</span>
      <span className={mono ? 'font-mono' : ''}>{value}</span>
    </div>
  );
}
