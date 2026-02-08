export function Header() {
  return (
    <div className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-5 py-3 bg-gradient-to-b from-[var(--color-bg)] to-transparent pointer-events-none">
      <div className="flex items-center gap-3 pointer-events-auto">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[var(--color-accent)] to-[#8b5cf6] flex items-center justify-center text-sm font-bold">
          P
        </div>
        <div>
          <h1 className="text-base font-bold tracking-tight">Panacea</h1>
          <p className="text-[10px] text-[var(--color-text-muted)] -mt-0.5">
            Orbital Debris Collision Prediction
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3 pointer-events-auto text-xs text-[var(--color-text-muted)]">
        <span className="px-2 py-0.5 rounded bg-[var(--color-surface-2)] border border-[var(--color-border)]">
          AIPI 540
        </span>
        <span>Live Data from CelesTrak</span>
      </div>
    </div>
  );
}
