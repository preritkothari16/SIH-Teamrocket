import type { Vessel } from '../types/schema';

export class VesselList {
  private container: HTMLElement;
  private onSelect: (mmsi: string) => void;

  constructor(container: HTMLElement, onSelect: (mmsi: string) => void) {
    this.container = container;
    this.onSelect = onSelect;
  }

  render(vessels: Vessel[]): void {
    if (vessels.length === 0) {
      this.container.innerHTML = `
        <div class="p-4 text-center text-sm text-zinc-500 dark:text-zinc-400">
          No vessel attribution data
        </div>
      `;
      return;
    }

    const items = vessels
      .sort((a, b) => b.score - a.score)
      .map((v) => {
        const scorePct = (v.score * 100).toFixed(0);
        const scoreColor = v.score >= 0.8 ? 'bg-emerald-500' : v.score >= 0.5 ? 'bg-amber-500' : 'bg-zinc-400';
        const time = new Date(v.cpa_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        return `
        <button class="vessel-item w-full text-left px-4 py-3 hover:bg-zinc-50 dark:hover:bg-zinc-800/50 border-b border-zinc-100 dark:border-zinc-800 transition-colors" data-mmsi="${v.mmsi}">
          <div class="flex items-center justify-between mb-1">
            <span class="text-sm font-medium text-zinc-900 dark:text-zinc-100 truncate">${v.name}</span>
            <span class="text-xs font-mono text-zinc-500 dark:text-zinc-400 ml-2">${v.mmsi}</span>
          </div>
          <div class="flex items-center gap-2 text-xs">
            <span class="px-1.5 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400">${v.vessel_type}</span>
            <div class="flex items-center gap-1">
              <div class="w-8 h-1.5 rounded-full bg-zinc-200 dark:bg-zinc-700 overflow-hidden">
                <div class="h-full rounded-full ${scoreColor}" style="width:${scorePct}%"></div>
              </div>
              <span class="text-zinc-500 dark:text-zinc-400">${scorePct}%</span>
            </div>
          </div>
          <div class="flex items-center gap-3 mt-1.5 text-xs text-zinc-500 dark:text-zinc-400">
            <span>CPA: ${v.cpa_distance_km.toFixed(1)} km</span>
            <span>at ${time}</span>
          </div>
          <p class="text-xs text-zinc-400 dark:text-zinc-500 mt-1 line-clamp-2">${v.explanation}</p>
        </button>
      `;
      })
      .join('');

    this.container.innerHTML = `
      <div class="p-2 border-b border-zinc-100 dark:border-zinc-800">
        <div class="flex items-center justify-between px-2">
          <h3 class="text-sm font-semibold text-zinc-900 dark:text-zinc-100">Vessels</h3>
          <span class="text-xs px-1.5 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400">${vessels.length}</span>
        </div>
      </div>
      <div class="overflow-y-auto max-h-[300px]">${items}</div>
    `;

    this.container.querySelectorAll<HTMLElement>('.vessel-item').forEach((el) => {
      el.addEventListener('click', () => {
        const mmsi = el.dataset.mmsi;
        if (mmsi) this.onSelect(mmsi);
      });
    });
  }

  destroy(): void {
    this.container.innerHTML = '';
  }
}