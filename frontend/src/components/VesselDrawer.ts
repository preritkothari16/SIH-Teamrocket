import type { Vessel } from '../types/schema';

export class VesselDrawer {
  private drawer: HTMLElement;
  private content: HTMLElement;
  private closeBtn: HTMLElement;
  constructor(drawer: HTMLElement) {
    this.drawer = drawer;
    this.content = drawer.querySelector('#vessel-drawer-content') as HTMLElement;
    this.closeBtn = drawer.querySelector('#vessel-drawer-close') as HTMLElement;

    this.closeBtn.addEventListener('click', () => this.hide());
  }

  render(vessels: Vessel[]): void {
    if (vessels.length === 0) {
      this.hide();
      return;
    }

    const items = vessels.map((vessel, i) => {
      const rank = i + 1;
      const rankClass = rank === 1 ? 'rank-1' : rank === 2 ? 'rank-2' : rank === 3 ? 'rank-3' : 'rank-default';
      const scorePct = (vessel.score * 100).toFixed(1);
      const scoreBarColor = vessel.score >= 0.7 ? 'bg-emerald-500' : vessel.score >= 0.4 ? 'bg-amber-500' : 'bg-red-500';
      const isTop = rank === 1;
      const cpaDist = vessel.cpa_distance_km.toFixed(1);
      const cpaTime = new Date(vessel.cpa_time).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
      return `
        <div class="px-4 py-3 border-b border-zinc-800/20 hover:bg-zinc-800/30 transition-colors group ${isTop ? 'bg-zinc-800/20' : ''}">
          <div class="flex items-start gap-3">
            <!-- Rank Badge -->
            <div class="rank-badge ${rankClass} shrink-0">${rank}</div>

            <!-- Vessel Info -->
            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-2 mb-1">
                <span class="text-sm font-semibold text-zinc-100 truncate">${vessel.name || vessel.mmsi}</span>
                ${isTop ? '<span class="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 font-bold uppercase">Top</span>' : ''}
              </div>

              <!-- MMSI + Type -->
              <div class="flex items-center gap-2 mb-2">
                <span class="text-[10px] text-zinc-500 font-mono">${vessel.mmsi}</span>
                <span class="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700/50 capitalize">${vessel.vessel_type}</span>
              </div>

              <!-- Score Bar -->
              <div class="flex items-center gap-2 mb-2">
                <div class="flex-1 h-2 rounded-full bg-zinc-800/60 overflow-hidden">
                  <div class="h-full rounded-full ${scoreBarColor} score-bar-fill" style="--fill-width:${scorePct}%; width:${scorePct}%"></div>
                </div>
                <span class="text-xs font-mono font-semibold ${vessel.score >= 0.7 ? 'text-emerald-400' : vessel.score >= 0.4 ? 'text-amber-400' : 'text-red-400'}">${scorePct}%</span>
              </div>

              <!-- CPA Info -->
              <div class="grid grid-cols-2 gap-2 mb-2">
                <div class="flex items-center gap-1.5 text-[11px]">
                  <svg class="w-3 h-3 text-zinc-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/><circle cx="12" cy="10" r="3"/></svg>
                  <span class="text-zinc-500">CPA:</span>
                  <span class="text-zinc-300 font-medium">${cpaDist} km</span>
                </div>
                <div class="flex items-center gap-1.5 text-[11px]">
                  <svg class="w-3 h-3 text-zinc-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                  <span class="text-zinc-500">At:</span>
                  <span class="text-zinc-300 font-medium">${cpaTime}</span>
                </div>
              </div>

              <!-- Explanation -->
              <div class="text-[11px] text-zinc-500 leading-relaxed">
                ${vessel.explanation}
              </div>
            </div>
          </div>
        </div>
      `;
    }).join('');

    this.content.innerHTML = items;
    this.show();
  }

  show(): void {
    this.drawer.style.transform = 'translateY(0)';
  }

  hide(): void {
    this.drawer.style.transform = 'translateY(100%)';
  }

  destroy(): void {
    this.content.innerHTML = '';
    this.hide();
  }
}
