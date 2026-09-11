import type { Region } from '../types/schema';

const ALL_ID = '';

export class RegionChips {
  private container: HTMLElement;
  private onSelect: (regionId: string | null) => void;
  private selectedId: string | null = null;

  constructor(container: HTMLElement, onSelect: (regionId: string | null) => void) {
    this.container = container;
    this.onSelect = onSelect;
  }

  render(regions: Region[], selectedId: string | null = this.selectedId): void {
    this.selectedId = selectedId;

    const chipClass = (active: boolean, disabled: boolean): string => {
      if (disabled) {
        return 'bg-ocean-deep/30 border-border-panel/50 text-txt-muted/50 cursor-not-allowed';
      }
      if (active) {
        return 'bg-accent-primary/90 border-accent-primary/50 text-white shadow-lg shadow-accent-primary/20';
      }
      return 'bg-ocean-deep/60 border-border-panel text-txt-primary hover:bg-ocean-deep hover:text-white';
    };

    const allChip = `
      <button
        class="region-chip text-xs px-3 py-1.5 rounded-lg border backdrop-blur transition-all duration-200 shrink-0 ${chipClass(selectedId === null, false)}"
        data-region-id="${ALL_ID}"
        title="Show every processed run"
      >All</button>
    `;

    const regionChips = regions.map((r) => {
      const pending = !r.scene_id;
      const active = r.id === selectedId;
      return `
        <button
          class="region-chip text-xs px-3 py-1.5 rounded-lg border backdrop-blur transition-all duration-200 shrink-0 ${chipClass(active, pending)}"
          data-region-id="${r.id}"
          ${pending ? 'disabled' : ''}
          title="${pending ? 'No demo data yet' : r.label}"
        >${r.label}</button>
      `;
    }).join('');

    this.container.innerHTML = allChip + regionChips;
    this.bindClicks();
  }

  private bindClicks(): void {
    this.container.querySelectorAll<HTMLButtonElement>('.region-chip:not(:disabled)').forEach((el) => {
      el.addEventListener('click', () => {
        const id = el.dataset.regionId || null;
        this.onSelect(id);
      });
    });
  }

  destroy(): void {
    this.container.innerHTML = '';
  }
}
