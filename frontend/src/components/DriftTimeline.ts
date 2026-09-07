import type { DriftForecast, DriftHindcast } from '../types/schema';

export class DriftTimeline {
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  render(forecast: DriftForecast, hindcast: DriftHindcast): void {
    const forecastItems = forecast.forecast.map((entry) => {
      const time = new Date(entry.time).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
      return `
        <div class="flex items-center gap-2 px-3 py-2 text-xs">
          <div class="w-2 h-2 rounded-full bg-amber-400 shrink-0"></div>
          <span class="text-zinc-500 dark:text-zinc-400 w-8 text-right">+${entry.hours}h</span>
          <span class="text-zinc-900 dark:text-zinc-100 font-mono">${time}</span>
        </div>
      `;
    }).join('');

    const hindcastItems = hindcast.hindcast.map((entry) => {
      const time = new Date(entry.time).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
      return `
        <div class="flex items-center gap-2 px-3 py-2 text-xs">
          <div class="w-2 h-2 rounded-full bg-zinc-400 shrink-0"></div>
          <span class="text-zinc-500 dark:text-zinc-400 w-8 text-right">ago</span>
          <span class="text-zinc-900 dark:text-zinc-100 font-mono">${time}</span>
        </div>
      `;
    }).join('');

    this.container.innerHTML = `
      <div class="p-2 border-b border-zinc-100 dark:border-zinc-800">
        <h3 class="text-sm font-semibold text-zinc-900 dark:text-zinc-100 px-2">Drift Timeline</h3>
      </div>
      <div class="max-h-[200px] overflow-y-auto">
        ${hindcastItems}
        ${forecastItems}
      </div>
    `;
  }

  destroy(): void {
    this.container.innerHTML = '';
  }
}