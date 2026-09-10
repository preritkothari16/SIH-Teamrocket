import type { PipelineRun, Provenance } from '../types/schema';

const SAR_SOURCE_LABELS: Record<string, string> = {
  cdse: 'Copernicus Data Space (CDSE)',
  local: 'Local disk',
};

function label(value: string | null | undefined, fallback: string, map?: Record<string, string>): string {
  if (!value) return fallback;
  return map?.[value] ?? value;
}

export class ProvenancePanel {
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  render(run: PipelineRun | null): void {
    if (!run || !run.provenance) {
      this.container.innerHTML = '';
      return;
    }

    const p: Provenance = run.provenance;
    const rows: Array<[string, string]> = [
      ['SAR', `${label(p.sar_source, 'unknown source', SAR_SOURCE_LABELS)}${p.sar_scene_id ? ` — ${p.sar_scene_id}` : ''}`],
      ['AIS', label(p.ais_source_label, 'no AIS export given')],
      ['Wind', label(p.wind_source, 'unavailable')],
      ['Current', label(p.current_source, 'unavailable')],
    ];

    this.container.innerHTML = `
      <div class="bg-zinc-900/95 backdrop-blur-xl border border-zinc-700/50 rounded-2xl overflow-hidden shadow-2xl shadow-black/40 animate-fade-in">
        <div class="p-3 border-b border-zinc-800/50">
          <span class="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">Data Provenance</span>
        </div>
        <div class="p-3 space-y-1.5">
          ${rows.map(([k, v]) => `
            <div class="flex items-center justify-between gap-3 py-1">
              <span class="text-[10px] text-zinc-500 uppercase tracking-wider">${k}</span>
              <span class="text-xs text-zinc-300 font-mono text-right truncate max-w-[65%]" title="${v}">${v}</span>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  destroy(): void {
    this.container.innerHTML = '';
  }
}
