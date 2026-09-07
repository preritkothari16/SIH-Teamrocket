import type { SpillObject } from '../types/schema';

export class SpillPanel {
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  render(spill: SpillObject | null): void {
    if (!spill) {
      this.container.innerHTML = `
        <div class="p-4 text-center text-sm text-zinc-500 dark:text-zinc-400">
          No spill data loaded
        </div>
      `;
      return;
    }

    const confidencePct = (spill.confidence * 100).toFixed(1);
    const confidenceColor = spill.confidence >= 0.8 ? 'text-emerald-500' : spill.confidence >= 0.5 ? 'text-amber-500' : 'text-red-500';
    const time = new Date(spill.acquisition_timestamp).toLocaleString();

    this.container.innerHTML = `
      <div class="p-4 space-y-3">
        <div class="flex items-center justify-between">
          <h3 class="text-sm font-semibold text-zinc-900 dark:text-zinc-100">Spill Details</h3>
          <span class="text-xs px-2 py-0.5 rounded-full bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400">${spill.scene_id}</span>
        </div>

        <div class="grid grid-cols-2 gap-2">
          <div class="rounded-lg bg-zinc-50 dark:bg-zinc-800/50 p-2">
            <div class="text-xs text-zinc-500 dark:text-zinc-400">Confidence</div>
            <div class="text-lg font-bold ${confidenceColor}">${confidencePct}%</div>
          </div>
          <div class="rounded-lg bg-zinc-50 dark:bg-zinc-800/50 p-2">
            <div class="text-xs text-zinc-500 dark:text-zinc-400">Area</div>
            <div class="text-lg font-bold text-zinc-900 dark:text-zinc-100">${spill.area_km2.toFixed(2)} <span class="text-xs font-normal">km²</span></div>
          </div>
        </div>

        <div class="space-y-2 text-xs">
          <div class="flex justify-between">
            <span class="text-zinc-500 dark:text-zinc-400">Acquired</span>
            <span class="text-zinc-900 dark:text-zinc-100 font-mono">${time}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-zinc-500 dark:text-zinc-400">Elongation</span>
            <span class="text-zinc-900 dark:text-zinc-100">${spill.elongation.toFixed(2)}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-zinc-500 dark:text-zinc-400">Bearing</span>
            <span class="text-zinc-900 dark:text-zinc-100">${spill.major_axis_bearing.toFixed(1)}°</span>
          </div>
          <div class="flex justify-between">
            <span class="text-zinc-500 dark:text-zinc-400">Centroid</span>
            <span class="text-zinc-900 dark:text-zinc-100 font-mono">${spill.centroid.lat.toFixed(4)}, ${spill.centroid.lon.toFixed(4)}</span>
          </div>
        </div>
      </div>
    `;
  }

  destroy(): void {
    this.container.innerHTML = '';
  }
}