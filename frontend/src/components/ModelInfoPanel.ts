import type { ModelInfo } from '../types/schema';

function pct(value: number | undefined): string {
  return value === undefined || Number.isNaN(value) ? '—' : `${(value * 100).toFixed(1)}%`;
}

function metricTile(label: string, value: string, color = 'text-zinc-100'): string {
  return `
    <div class="rounded-xl bg-zinc-800/40 p-3 border border-zinc-700/30">
      <div class="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">${label}</div>
      <div class="text-lg font-bold font-mono ${color}">${value}</div>
    </div>
  `;
}

export class ModelInfoPanel {
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  render(info: ModelInfo | null): void {
    if (!info) {
      this.container.innerHTML = `
        <div class="p-6 text-center text-xs text-zinc-500">Loading model info&hellip;</div>
      `;
      return;
    }

    if (!info.trained) {
      this.container.innerHTML = `
        <div class="p-4 border-b border-zinc-800/50">
          <span class="text-xs font-bold text-zinc-300 uppercase tracking-widest">Model Card</span>
        </div>
        <div class="p-6 text-center">
          <svg class="w-10 h-10 mx-auto text-zinc-700 mb-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <rect x="3" y="3" width="18" height="18" rx="2"/>
            <path d="M9 9h.01M15 9h.01M9 15c1-1 5-1 6 0"/>
          </svg>
          <p class="text-sm font-medium text-zinc-300 mb-1">Model not yet trained</p>
          <p class="text-xs text-zinc-500 leading-relaxed">
            No checkpoint exists here yet &mdash; <code class="text-zinc-400">models/</code> is empty.
            This panel fills in on its own the first time a real training run
            produces <code class="text-zinc-400">models/best.pt</code>.
          </p>
        </div>
      `;
      return;
    }

    const arch = info.architecture;

    this.container.innerHTML = `
      <div class="p-4 border-b border-zinc-800/50">
        <span class="text-xs font-bold text-zinc-300 uppercase tracking-widest">Model Card</span>
      </div>
      <div class="p-4 space-y-4">
        ${arch ? `
          <div>
            <div class="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">Architecture</div>
            <div class="text-sm text-zinc-100 font-mono">${arch.arch} / ${arch.encoder}</div>
            <div class="text-xs text-zinc-500 mt-1">${arch.num_classes} classes &middot; ${arch.image_size}px tiles &middot; ${arch.in_channels} ch</div>
          </div>
        ` : ''}

        <div class="grid grid-cols-2 gap-3">
          ${metricTile('Train / Val Samples', `${info.train_samples ?? '—'} / ${info.val_samples ?? '—'}`)}
          ${metricTile('Mean IoU', pct(info.mean_iou), 'text-emerald-400')}
        </div>

        <div class="grid grid-cols-2 gap-3">
          ${metricTile('Oil IoU', pct(info.oil_iou), 'text-emerald-400')}
          ${metricTile('Look-alike IoU', pct(info.look_alike_iou), 'text-amber-400')}
        </div>

        <div>
          <div class="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Oil class</div>
          <div class="grid grid-cols-3 gap-2">
            ${metricTile('Precision', pct(info.precision))}
            ${metricTile('Recall', pct(info.recall))}
            ${metricTile('Dice', pct(info.dice))}
          </div>
        </div>

        <div>
          <div class="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Oil &harr; Look-alike Confusion</div>
          <div class="space-y-1.5">
            <div class="flex items-center justify-between py-1 border-b border-zinc-800/30">
              <span class="text-xs text-zinc-500">Oil called look-alike</span>
              <span class="text-xs text-zinc-300 font-mono">${pct(info.oil_as_lookalike_rate)}</span>
            </div>
            <div class="flex items-center justify-between py-1">
              <span class="text-xs text-zinc-500">Look-alike called oil</span>
              <span class="text-xs text-zinc-300 font-mono">${pct(info.lookalike_as_oil_rate)}</span>
            </div>
          </div>
        </div>

        ${info.trained_at ? `
          <div class="text-[10px] text-zinc-600 pt-1">
            Trained ${new Date(info.trained_at).toLocaleString()}
          </div>
        ` : ''}
      </div>
    `;
  }

  destroy(): void {
    this.container.innerHTML = '';
  }
}
