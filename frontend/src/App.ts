import { MapView } from './components/MapView';
import { SpillPanel } from './components/SpillPanel';
import { VesselList } from './components/VesselList';
import { DriftTimeline } from './components/DriftTimeline';
import { AlertBadge } from './components/AlertBadge';
import { listRuns, getRun } from './api/client';
import type { PipelineRun } from './types/schema';

export class App {
  private map: MapView;
  private spillPanel: SpillPanel;
  private vesselList: VesselList;
  private driftTimeline: DriftTimeline;
  private alertBadge: AlertBadge;
  private apiAvailable = false;

  constructor() {
    const mapEl = document.getElementById('map')!;
    const spillEl = document.getElementById('spill-panel')!;
    const vesselEl = document.getElementById('vessel-list')!;
    const driftEl = document.getElementById('drift-timeline')!;
    const alertEl = document.getElementById('alert-badge')!;

    this.map = new MapView(mapEl);
    this.spillPanel = new SpillPanel(spillEl);
    this.vesselList = new VesselList(vesselEl, (mmsi: string) => this.map.highlightVessel(mmsi));
    this.driftTimeline = new DriftTimeline(driftEl);
    this.alertBadge = new AlertBadge(alertEl);

    this.init();
  }

  private async init(): Promise<void> {
    await this.checkApi();
    this.populateDropdown();
    this.setupSelector();
    this.setupLegendToggle();
    this.setupClearButton();
  }

  private async checkApi(): Promise<void> {
    try {
      const res = await fetch('/api/health');
      this.apiAvailable = res.ok;
    } catch {
      this.apiAvailable = false;
    }
  }

  private populateDropdown(): void {
    const select = document.getElementById('mock-select') as HTMLSelectElement;
    if (!select) return;

    if (this.apiAvailable) {
      // Fetch real runs from the API and populate the dropdown
      listRuns().then((runs) => {
        if (runs.length > 0) {
          // Add a separator comment
          const optgroup = document.createElement('optgroup');
          optgroup.label = 'Backend runs';
          runs.forEach((run) => {
            const opt = document.createElement('option');
            opt.value = `api:${run.scene_id}`;
            opt.textContent = `${run.scene_id} (${run.area_km2.toFixed(1)} km²)`;
            optgroup.appendChild(opt);
          });
          select.insertBefore(optgroup, select.firstChild);
        }
      }).catch(() => {
        // API not reachable — mock options remain
      });
    }
  }

  private setupSelector(): void {
    const select = document.getElementById('mock-select') as HTMLSelectElement;
    if (!select) return;

    select.addEventListener('change', async () => {
      const value = select.value;
      if (!value) {
        this.clear();
        return;
      }

      if (value.startsWith('api:')) {
        // Real backend run
        const sceneId = value.slice(4);
        try {
          const data = await getRun(sceneId);
          this.loadRun(data);
        } catch (err) {
          console.error('Failed to load run from API:', err);
        }
      } else {
        // Mock fixture
        try {
          const response = await fetch(`/src/mocks/fixtures/${value}.json`);
          const data: PipelineRun = await response.json();
          this.loadRun(data);
        } catch (err) {
          console.error('Failed to load mock data:', err);
        }
      }
    });
  }

  private setupLegendToggle(): void {
    const btn = document.getElementById('legend-toggle');
    const legend = document.getElementById('legend');
    if (!btn || !legend) return;

    btn.addEventListener('click', () => {
      legend.classList.toggle('hidden');
    });
  }

  private setupClearButton(): void {
    const btn = document.getElementById('clear-btn');
    if (!btn) return;

    btn.addEventListener('click', () => {
      this.clear();
      const select = document.getElementById('mock-select') as HTMLSelectElement;
      if (select) select.value = '';
    });
  }

  loadRun(run: PipelineRun): void {
    this.map.setSpill(run.spill);
    this.spillPanel.render(run.spill);
    this.vesselList.render(run.vessels);
    this.driftTimeline.render(run.drift, run.drift);
    this.alertBadge.render(run.alert);
  }

  clear(): void {
    this.map.clear();
    this.spillPanel.render(null);
    this.vesselList.render([]);
    this.driftTimeline.render({ forecast: [] }, { hindcast: [] });
    this.alertBadge.render(null);
  }

  destroy(): void {
    this.map.destroy();
    this.spillPanel.destroy();
    this.vesselList.destroy();
    this.driftTimeline.destroy();
    this.alertBadge.destroy();
  }
}