import type { Vessel } from '../types/schema';

const SCORE_COLORS: Record<string, string> = {
  high: 'bg-emerald-500',
  mid: 'bg-amber-500',
  low: 'bg-zinc-500',
};

function scoreColor(score: number): string {
  if (score >= 0.8) return SCORE_COLORS.high;
  if (score >= 0.5) return SCORE_COLORS.mid;
  return SCORE_COLORS.low;
}

export class VesselDrawer {
  private drawerEl: HTMLElement;
  private contentEl: HTMLElement;
  private closeBtn: HTMLElement;
  constructor(drawerEl: HTMLElement, contentEl: HTMLElement, closeBtn: HTMLElement) {
    this.drawerEl = drawerEl;
    this.contentEl = contentEl;
    this.closeBtn = closeBtn;
    this.closeBtn.addEventListener('click', () => this.close());
  }

  render(vessels: Vessel[]): void {
    if (vessels.length === 0) {
      this.contentEl.innerHTML = `
        <div class="px-4 py-3 text-xs text-zinc-500 text-center">
          No vessel attribution data
        </div>
      `;
      this.open();
      return;
    }

    const items = vessels
      .sort((a, b) => b.score - a.score)
      .map((v, i) => {
        const scorePct = (v.score * 100).toFixed(0);
        const color = scoreColor(v.score);
        const time = new Date(v.cpa_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        return `
          <div class="flex items-start gap-3 px-4 py-2.5 border-b border-zinc-800/50">
            <div class="w-5 h-5 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center shrink-0 mt-0.5">
              <span class="text-xs font-medium text-zinc-400">${i + 1}</span>
            </div>
            <div class="flex-1 min-w-0">
              <div class="flex items-center justify-between gap-2">
                <span class="text-sm font-medium text-zinc-200 truncate">${v.name}</span>
                <span class="text-xs text-zinc-500 font-mono shrink-0">${v.mmsi}</span>
              </div>
              <div class="flex items-center gap-2 mt-1 text-xs text-zinc-500">
                <span class="px-1 py-0.5 rounded bg-zinc-800 text-zinc-400">${v.vessel_type}</span>
                <div class="flex items-center gap-1">
                  <div class="w-8 h-1.5 rounded-full bg-zinc-700 overflow-hidden">
                    <div class="h-full rounded-full ${color}" style="width:${scorePct}%"></div>
                  </div>
                  <span>${scorePct}%</span>
                </div>
                <span>CPA ${v.cpa_distance_km.toFixed(1)} km</span>
                <span>at ${time}</span>
              </div>
              <p class="text-xs text-zinc-500 mt-1 line-clamp-2">${v.explanation}</p>
            </div>
          </div>
        `;
      })
      .join('');

    this.contentEl.innerHTML = items;
    this.open();
  }

  private open(): void {
    this.drawerEl.style.transform = 'translateY(0)';
  }

  close(): void {
    this.drawerEl.style.transform = 'translateY(100%)';
  }

  destroy(): void {
    this.contentEl.innerHTML = '';
    this.close();
  }
}
