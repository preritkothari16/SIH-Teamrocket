import type { PipelineRun } from '../types/schema';

const STATUS_COLORS: Record<string, string> = {
  new: 'bg-red-500',
  update: 'bg-orange-400',
  possible: 'bg-yellow-400',
  none: 'bg-zinc-500',
};

const STATUS_LABELS: Record<string, string> = {
  new: 'New',
  update: 'Update',
  possible: 'Possible',
  none: 'None',
};

export class RunList {
  private container: HTMLElement;
  private onSelect: (runId: string | null) => void;
  private selectedId: string | null = null;

  constructor(container: HTMLElement, onSelect: (runId: string | null) => void) {
    this.container = container;
    this.onSelect = onSelect;
  }

  render(runs: PipelineRun[]): void {
    if (runs.length === 0) {
      this.container.innerHTML = `
        <div class="p-4 text-center text-xs text-zinc-500">
          No spills detected
        </div>
      `;
      return;
    }

    const items = runs.map((run, i) => {
      const id = run.alert?.spill_id ?? run.spill.scene_id;
      const status = run.alert?.status ?? 'none';
      const color = STATUS_COLORS[status] ?? STATUS_COLORS.none;
      const label = STATUS_LABELS[status] ?? status;
      const isSelected = id === this.selectedId;
      const vesselCount = run.vessels.length;

      return `
        <button class="run-item w-full text-left px-3 py-3 border-b border-zinc-800/50 hover:bg-zinc-800/50 transition-colors ${isSelected ? 'bg-zinc-800/70 ring-1 ring-inset ring-blue-500/30' : ''}" data-run-id="${id}" data-index="${i}">
          <div class="flex items-center gap-2 mb-1.5">
            <div class="w-2 h-2 rounded-full ${color} shrink-0"></div>
            <span class="text-xs font-medium text-zinc-200 truncate">${id}</span>
          </div>
          <div class="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-zinc-500">
            <span class="whitespace-nowrap">${run.spill.area_km2.toFixed(1)} km²</span>
            <span class="whitespace-nowrap">${(run.spill.confidence * 100).toFixed(0)}% conf</span>
            <span class="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 whitespace-nowrap">${label}</span>
            ${vesselCount > 0 ? `<span class="whitespace-nowrap">${vesselCount} vessel${vesselCount > 1 ? 's' : ''}</span>` : ''}
          </div>
        </button>
      `;
    }).join('');

    this.container.innerHTML = items;
    this.bindClicks();
  }

  setSelected(runId: string | null): void {
    this.selectedId = runId;
    this.container.querySelectorAll<HTMLElement>('.run-item').forEach((el) => {
      const id = el.dataset.runId;
      const isSelected = id === runId;
      el.classList.toggle('bg-zinc-800/70', isSelected);
      el.classList.toggle('ring-1', isSelected);
      el.classList.toggle('ring-inset', isSelected);
      el.classList.toggle('ring-blue-500/30', isSelected);
    });
  }

  private bindClicks(): void {
    this.container.querySelectorAll<HTMLElement>('.run-item').forEach((el) => {
      el.addEventListener('click', () => {
        const id = el.dataset.runId ?? null;
        // Toggle deselect if clicking the same run
        const newId = id === this.selectedId ? null : id;
        this.onSelect(newId);
      });
    });
  }

  destroy(): void {
    this.container.innerHTML = '';
  }
}
