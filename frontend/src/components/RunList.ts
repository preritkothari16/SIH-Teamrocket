import type { PipelineRun } from '../types/schema';
import { escapeHtml } from '../utils/escape';

const STATUS_COLORS: Record<string, string> = {
  new: 'bg-red-500',
  update: 'bg-orange-400',
  possible: 'bg-yellow-400',
  none: 'bg-zinc-500',
};

const STATUS_GLOW: Record<string, string> = {
  new: 'shadow-red-500/50',
  update: 'shadow-orange-400/50',
  possible: 'shadow-yellow-400/50',
  none: '',
};

const STATUS_LABELS: Record<string, string> = {
  new: 'New',
  update: 'Update',
  possible: 'Possible',
  none: 'None',
};

const STATUS_TEXT: Record<string, string> = {
  new: 'text-red-400',
  update: 'text-orange-400',
  possible: 'text-yellow-400',
  none: 'text-zinc-500',
};

const STATUS_BG: Record<string, string> = {
  new: 'bg-red-900/40 border-red-700/50',
  update: 'bg-orange-900/40 border-orange-700/50',
  possible: 'bg-yellow-900/40 border-yellow-700/50',
  none: 'bg-zinc-800/50 border-zinc-700/40',
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
    // Update stats
    const totalEl = document.getElementById('stat-total');
    const alertedEl = document.getElementById('stat-alerted');
    const areaEl = document.getElementById('stat-area');
    const countEl = document.getElementById('spill-count');
    if (totalEl) totalEl.textContent = String(runs.length);
    // Same predicate as App.ts::updateStatsStrip's #strip-active — these two
    // widgets used to disagree (this one skipped 'update'); unified so
    // "Active" in the sidebar and the HUD strip always mean the same count.
    if (alertedEl) alertedEl.textContent = String(runs.filter(r => !!r.alert?.status && r.alert.status !== 'none').length);
    if (areaEl) areaEl.textContent = runs.reduce((s, r) => s + r.spill.area_km2, 0).toFixed(1);
    if (countEl) countEl.textContent = String(runs.length);

    if (runs.length === 0) {
      this.container.innerHTML = `
        <div class="p-6 text-center">
          <svg class="w-8 h-8 mx-auto text-zinc-700 mb-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M9 12l2 2 4-4"/>
            <circle cx="12" cy="12" r="10"/>
          </svg>
          <p class="text-xs text-zinc-500 font-medium">No spills detected</p>
          <p class="text-[10px] text-zinc-600 mt-1">Run the pipeline to see results</p>
        </div>
      `;
      return;
    }

    const items = runs.map((run, i) => {
      const id = run.alert?.spill_id || run.spill.scene_id;
      const status = run.alert?.status ?? 'none';
      const color = STATUS_COLORS[status] ?? STATUS_COLORS.none;
      const glow = STATUS_GLOW[status] ?? '';
      const label = STATUS_LABELS[status] ?? status;
      const textColor = STATUS_TEXT[status] ?? STATUS_TEXT.none;
      const bgColor = STATUS_BG[status] ?? STATUS_BG.none;
      const isSelected = id === this.selectedId;
      const vesselCount = run.vessels.length;
      const conf = run.spill.confidence;
      const confPct = (conf * 100).toFixed(0);
      const confColor = conf >= 0.8 ? 'text-emerald-400' : conf >= 0.5 ? 'text-amber-400' : 'text-red-400';
      const confBarColor = conf >= 0.8 ? 'bg-emerald-400' : conf >= 0.5 ? 'bg-amber-400' : 'bg-red-400';

      return `
        <button class="run-item w-full text-left px-4 py-3.5 border-b border-zinc-800/30 hover:bg-zinc-800/40 transition-all duration-200 group card-hover ${isSelected ? 'bg-zinc-800/60 border-l-2 border-l-blue-500' : ''}" data-run-id="${escapeHtml(id)}" data-index="${i}">
          <div class="flex items-start gap-3">
            <div class="mt-1 relative">
              <div class="w-3.5 h-3.5 rounded-full ${color} shadow-lg ${glow} shrink-0 ${status === 'new' ? 'glow-pulse' : ''}"></div>
            </div>
            <div class="flex-1 min-w-0">
              <div class="flex items-center justify-between gap-2 mb-1.5">
                <span class="text-sm font-semibold text-zinc-100 truncate group-hover:text-white">${escapeHtml(id)}</span>
                <span class="text-[10px] px-2 py-0.5 rounded-full border font-semibold ${bgColor} ${textColor} shrink-0">${label}</span>
              </div>
              <div class="flex items-center gap-3 mb-2">
                <div class="flex items-center gap-1.5">
                  <div class="w-14 h-1.5 rounded-full bg-zinc-800 overflow-hidden">
                    <div class="h-full rounded-full ${confBarColor} transition-all duration-500" style="width:${confPct}%"></div>
                  </div>
                  <span class="text-xs ${confColor} font-mono font-medium">${confPct}%</span>
                </div>
                <span class="text-xs text-zinc-500 font-mono">${run.spill.area_km2.toFixed(1)} km&sup2;</span>
              </div>
              <div class="flex items-center gap-3 text-[10px] text-zinc-500">
                <span class="flex items-center gap-1">
                  <svg class="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                  ${new Date(run.spill.acquisition_timestamp).toLocaleDateString()}
                </span>
                ${vesselCount > 0 ? `
                  <span class="flex items-center gap-1 text-blue-400">
                    <svg class="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 17h1m0 0a2 2 0 104 0m-4 0a2 2 0 114 0m6-4V5a2 2 0 00-2-2H9a2 2 0 00-2 2v8m10 0H7"/></svg>
                    ${vesselCount} vessel${vesselCount > 1 ? 's' : ''}
                  </span>
                ` : ''}
              </div>
            </div>
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
      el.classList.toggle('bg-zinc-800/60', isSelected);
      el.classList.toggle('border-l-2', isSelected);
      el.classList.toggle('border-l-blue-500', isSelected);
    });
  }

  private bindClicks(): void {
    this.container.querySelectorAll<HTMLElement>('.run-item').forEach((el) => {
      el.addEventListener('click', () => {
        const id = el.dataset.runId ?? null;
        const newId = id === this.selectedId ? null : id;
        this.onSelect(newId);
      });
    });
  }

  destroy(): void {
    this.container.innerHTML = '';
  }
}
