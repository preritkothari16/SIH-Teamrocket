import type { PipelineRun } from '../types/schema';
import { escapeHtml } from '../utils/escape';

const STATUS_STYLES: Record<string, string> = {
  new: 'bg-red-900/50 text-red-300 border-red-700/60 shadow-red-900/30',
  update: 'bg-orange-900/50 text-orange-300 border-orange-700/60 shadow-orange-900/30',
  possible: 'bg-yellow-900/50 text-yellow-300 border-yellow-700/60 shadow-yellow-900/30',
  none: 'bg-zinc-800/60 text-zinc-400 border-zinc-700/50',
};

const STATUS_LABELS: Record<string, string> = {
  new: 'New Spill',
  update: 'Update',
  possible: 'Possible',
  none: 'No Alert',
};

const STATUS_DOT: Record<string, string> = {
  new: 'bg-red-500',
  update: 'bg-orange-400',
  possible: 'bg-yellow-400',
  none: 'bg-zinc-500',
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
    const statusDot = STATUS_DOT[status] ?? STATUS_DOT.none;
    const rules = alert?.rules_fired ?? [];
    const vesselCount = run.vessels.length;
    const drift = run.drift;
    const hasDrift = drift?.forecast && drift.forecast.length > 0;

    const conf = spill.confidence;
    const confPct = (conf * 100).toFixed(1);
    const confColor = conf >= 0.8 ? 'text-emerald-400' : conf >= 0.5 ? 'text-amber-400' : 'text-red-400';
    const confPctNum = Math.round(conf * 100);

    this.container.innerHTML = `
      <div class="bg-zinc-900/95 backdrop-blur-xl border border-zinc-700/50 rounded-2xl overflow-hidden shadow-2xl shadow-black/40 animate-fade-in">
        <!-- Status Header -->
        <div class="p-4 border-b border-zinc-800/50 flex items-center justify-between">
          <div class="flex items-center gap-2.5">
            <div class="w-2.5 h-2.5 rounded-full ${statusDot} ${status === 'new' ? 'glow-pulse' : ''}"></div>
            <span class="text-sm font-bold border px-2.5 py-0.5 rounded-lg ${statusStyle} shadow-lg">
              ${statusLabel}
            </span>
          </div>
          <span class="text-[10px] text-zinc-500 font-mono">${escapeHtml(spill.scene_id)}</span>
        </div>

        <div class="p-4 space-y-4">
          <!-- Confidence + Area -->
          <div class="grid grid-cols-2 gap-3">
            <div class="rounded-xl bg-zinc-800/40 p-3 border border-zinc-700/30">
              <div class="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">Confidence</div>
              <div class="flex items-end gap-2">
                <span class="text-xl font-bold ${confColor} font-mono">${confPct}%</span>
                <div class="w-12 h-1.5 rounded-full bg-zinc-700/50 overflow-hidden mb-1.5">
                  <div class="h-full rounded-full ${conf >= 0.8 ? 'bg-emerald-400' : conf >= 0.5 ? 'bg-amber-400' : 'bg-red-400'}" style="width:${confPctNum}%"></div>
                </div>
              </div>
            </div>
            <div class="rounded-xl bg-zinc-800/40 p-3 border border-zinc-700/30">
              <div class="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">Area</div>
              <div class="flex items-end gap-1">
                <span class="text-xl font-bold text-zinc-100 font-mono">${spill.area_km2.toFixed(2)}</span>
                <span class="text-xs text-zinc-500 mb-1">km&sup2;</span>
              </div>
            </div>
          </div>

          <!-- Properties -->
          <div class="space-y-2">
            <div class="flex items-center justify-between py-1.5 border-b border-zinc-800/30">
              <span class="text-xs text-zinc-500 flex items-center gap-1.5">
                <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/><circle cx="12" cy="10" r="3"/></svg>
                Centroid
              </span>
              <span class="text-xs text-zinc-300 font-mono">${spill.centroid.lat.toFixed(4)}, ${spill.centroid.lon.toFixed(4)}</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-zinc-800/30">
              <span class="text-xs text-zinc-500 flex items-center gap-1.5">
                <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 12h18"/></svg>
                Elongation
              </span>
              <span class="text-xs text-zinc-300 font-mono">${spill.elongation.toFixed(2)}x</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-zinc-800/30">
              <span class="text-xs text-zinc-500 flex items-center gap-1.5">
                <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 2v4m0 12v4M2 12h4m12 0h4"/></svg>
                Bearing
              </span>
              <span class="text-xs text-zinc-300 font-mono">${spill.major_axis_bearing.toFixed(1)}&deg;</span>
            </div>
            <div class="flex items-center justify-between py-1.5">
              <span class="text-xs text-zinc-500 flex items-center gap-1.5">
                <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                Acquired
              </span>
              <span class="text-xs text-zinc-300 font-mono">${new Date(spill.acquisition_timestamp).toLocaleString()}</span>
            </div>
          </div>

          <!-- Rules Fired -->
          ${rules.length > 0 ? `
            <div>
              <div class="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Alert Rules</div>
              <div class="flex flex-wrap gap-1.5">
                ${rules.map((r) => `<span class="rule-tag rule-failed">${escapeHtml(r)}</span>`).join('')}
              </div>
            </div>
          ` : `
            <div class="flex items-center gap-2 py-2 px-3 rounded-lg bg-emerald-900/20 border border-emerald-800/30">
              <svg class="w-4 h-4 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
              <span class="text-xs text-emerald-400 font-medium">All rules passed</span>
            </div>
          `}

          <!-- Drift Forecast Summary -->
          ${hasDrift ? `
            <div>
              <div class="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Drift Forecast</div>
              <div class="flex gap-2">
                ${drift!.forecast!.slice(0, 4).map((f: any) => `
                  <div class="flex-1 text-center p-2 rounded-lg bg-zinc-800/40 border border-zinc-700/30">
                    <div class="text-xs font-bold text-amber-400 font-mono">+${f.hours}h</div>
                    <div class="text-[10px] text-zinc-500 mt-0.5">${new Date(f.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
                  </div>
                `).join('')}
              </div>
            </div>
          ` : ''}

          <!-- Vessels Footer -->
          <div class="flex items-center justify-between py-2.5 px-3 rounded-xl bg-zinc-800/30 border border-zinc-700/30">
            <span class="text-xs text-zinc-400 flex items-center gap-1.5">
              <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 17h1m0 0a2 2 0 104 0m-4 0a2 2 0 114 0m6-4V5a2 2 0 00-2-2H9a2 2 0 00-2 2v8m10 0H7"/></svg>
              Vessels
            </span>
            <span class="text-xs font-semibold ${vesselCount > 0 ? 'text-blue-400' : 'text-zinc-500'}">
              ${vesselCount > 0 ? `${vesselCount} candidate${vesselCount > 1 ? 's' : ''}` : 'No attribution'}
            </span>
          </div>
        </div>
      </div>
    `;
  }

  destroy(): void {
    this.container.innerHTML = '';
  }
}
