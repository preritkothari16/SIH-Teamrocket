import type { PipelineRun } from '../types/schema';

const STATUS_STYLES: Record<string, string> = {
  new: 'bg-red-900/40 text-red-300 border-red-700/50',
  update: 'bg-orange-900/40 text-orange-300 border-orange-700/50',
  possible: 'bg-yellow-900/40 text-yellow-300 border-yellow-700/50',
  none: 'bg-zinc-800 text-zinc-400 border-zinc-700',
};

const STATUS_LABELS: Record<string, string> = {
  new: 'New Spill',
  update: 'Update',
  possible: 'Possible',
  none: 'No Alert',
};

export class SpillOverlay {
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  render(run: PipelineRun | null): void {
    if (!run) {
      this.container.innerHTML = '';
      return;
    }

    const spill = run.spill;
    const alert = run.alert;
    const status = alert?.status ?? 'none';
    const statusStyle = STATUS_STYLES[status] ?? STATUS_STYLES.none;
    const statusLabel = STATUS_LABELS[status] ?? status;
    const rules = alert?.rules_fired ?? [];
    const vesselCount = run.vessels.length;

    this.container.innerHTML = `
      <div class="bg-zinc-900/95 backdrop-blur border border-zinc-700 rounded-lg overflow-hidden">
        <div class="p-3 border-b border-zinc-800 flex items-center justify-between">
          <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${statusStyle}">
            ${statusLabel}
          </span>
          <span class="text-xs text-zinc-500 font-mono">${spill.scene_id}</span>
        </div>
        <div class="p-3 space-y-2.5">
          <div class="grid grid-cols-2 gap-2">
            <div class="rounded bg-zinc-800/50 p-2">
              <div class="text-xs text-zinc-500">Confidence</div>
              <div class="text-base font-bold ${spill.confidence >= 0.8 ? 'text-emerald-400' : spill.confidence >= 0.5 ? 'text-amber-400' : 'text-red-400'}">
                ${(spill.confidence * 100).toFixed(1)}%
              </div>
            </div>
            <div class="rounded bg-zinc-800/50 p-2">
              <div class="text-xs text-zinc-500">Area</div>
              <div class="text-base font-bold text-zinc-100">
                ${spill.area_km2.toFixed(2)} <span class="text-xs font-normal text-zinc-400">km²</span>
              </div>
            </div>
          </div>
          <div class="space-y-1.5 text-xs">
            <div class="flex justify-between">
              <span class="text-zinc-500">Centroid</span>
              <span class="text-zinc-300 font-mono">${spill.centroid.lat.toFixed(4)}, ${spill.centroid.lon.toFixed(4)}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-zinc-500">Elongation</span>
              <span class="text-zinc-300">${spill.elongation.toFixed(2)}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-zinc-500">Bearing</span>
              <span class="text-zinc-300">${spill.major_axis_bearing.toFixed(1)}°</span>
            </div>
            <div class="flex justify-between">
              <span class="text-zinc-500">Acquired</span>
              <span class="text-zinc-300 font-mono">${new Date(spill.acquisition_timestamp).toLocaleString()}</span>
            </div>
          </div>
          ${rules.length > 0 ? `
            <div>
              <div class="text-xs text-zinc-500 mb-1">Rules fired</div>
              <div class="flex flex-wrap gap-1">
                ${rules.map((r) => `<span class="text-xs px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700">${r}</span>`).join('')}
              </div>
            </div>
          ` : ''}
          <div class="flex justify-between text-xs pt-1 border-t border-zinc-800">
            <span class="text-zinc-500">Vessels</span>
            <span class="text-zinc-300">${vesselCount > 0 ? `${vesselCount} candidate${vesselCount > 1 ? 's' : ''}` : 'No attribution data'}</span>
          </div>
        </div>
      </div>
    `;
  }

  destroy(): void {
    this.container.innerHTML = '';
  }
}
