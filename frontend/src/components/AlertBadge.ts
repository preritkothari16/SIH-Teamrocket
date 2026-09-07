import type { Alert } from '../types/schema';

const STATUS_STYLES: Record<Alert['status'], string> = {
  new: 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800',
  update: 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-800',
  possible: 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-800',
  none: 'bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 border-zinc-200 dark:border-zinc-700',
};

const STATUS_LABELS: Record<Alert['status'], string> = {
  new: 'New Spill',
  update: 'Update',
  possible: 'Possible',
  none: 'None',
};

export class AlertBadge {
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  render(alert: Alert | null): void {
    if (!alert) {
      this.container.innerHTML = '';
      return;
    }

    const style = STATUS_STYLES[alert.status];
    const label = STATUS_LABELS[alert.status];
    const firstSeen = new Date(alert.first_seen).toLocaleString();
    const rules = alert.rules_fired.map((r) => `<span class="text-xs px-1.5 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400">${r}</span>`).join(' ');

    this.container.innerHTML = `
      <div class="p-4 space-y-3">
        <div class="flex items-center gap-2">
          <span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold border ${style}">
            ${label}
          </span>
          <span class="text-xs text-zinc-500 dark:text-zinc-400">ID: ${alert.spill_id}</span>
        </div>

        <div class="space-y-2 text-xs">
          <div class="flex justify-between">
            <span class="text-zinc-500 dark:text-zinc-400">First seen</span>
            <span class="text-zinc-900 dark:text-zinc-100 font-mono">${firstSeen}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-zinc-500 dark:text-zinc-400">Last updated</span>
            <span class="text-zinc-900 dark:text-zinc-100 font-mono">${new Date(alert.last_updated).toLocaleString()}</span>
          </div>
        </div>

        ${alert.rules_fired.length > 0 ? `
          <div>
            <div class="text-xs text-zinc-500 dark:text-zinc-400 mb-1">Rules fired</div>
            <div class="flex flex-wrap gap-1">${rules}</div>
          </div>
        ` : ''}
      </div>
    `;
  }

  destroy(): void {
    this.container.innerHTML = '';
  }
}