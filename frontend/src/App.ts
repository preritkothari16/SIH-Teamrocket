import { SpillGlobe } from './components/ui/SpillGlobe';
import { RunList } from './components/RunList';
import { SpillOverlay } from './components/SpillOverlay';
import { ProvenancePanel } from './components/ProvenancePanel';
import { RegionChips } from './components/RegionChips';
import { ModelInfoPanel } from './components/ModelInfoPanel';
import { VesselDrawer } from './components/VesselDrawer';
import { listRuns, listRegions, getModelInfo, getRun, getReport } from './api/client';
import type { PipelineRun, Region } from './types/schema';

const USE_MOCKS = import.meta.env.VITE_USE_MOCKS === 'true';

export class App {
  private globe: SpillGlobe;
  private runList: RunList;
  private spillOverlay: SpillOverlay;
  private provenancePanel: ProvenancePanel;
  private regionChips: RegionChips;
  private modelInfoPanel: ModelInfoPanel;
  private modelInfoToggle: HTMLElement;
  private modelInfoPanelEl: HTMLElement;
  private vesselDrawer: VesselDrawer;
  private allRuns: PipelineRun[] = [];
  private regions: Region[] = [];
  private selectedRegionId: string | null = null;

  private runListPanel: HTMLElement;
  private runListToggle: HTMLElement;
  private detailStack: HTMLElement;
  private globeHint: HTMLElement;
  private loadingEl: HTMLElement;
  private errorEl: HTMLElement;
  private reportBtn: HTMLButtonElement;
  private selectedSceneId: string | null = null;

  constructor() {
    const globeCanvas = document.getElementById('globe-canvas') as HTMLCanvasElement;
    const runListEl = document.getElementById('run-list')!;
    const overlayEl = document.getElementById('spill-overlay')!;
    const provenanceEl = document.getElementById('provenance-panel')!;
    const regionChipsEl = document.getElementById('region-chips')!;
    const drawerEl = document.getElementById('vessel-drawer')!;
    this.modelInfoPanelEl = document.getElementById('model-info-panel')!;
    this.modelInfoToggle = document.getElementById('model-info-toggle')!;

    this.runListPanel = document.getElementById('run-list-panel')!;
    this.runListToggle = document.getElementById('run-list-toggle')!;
    this.detailStack = document.getElementById('detail-stack')!;
    this.globeHint = document.getElementById('globe-hint')!;
    this.loadingEl = document.getElementById('loading-state')!;
    this.errorEl = document.getElementById('error-state')!;
    this.reportBtn = document.getElementById('report-btn') as HTMLButtonElement;

    this.globe = new SpillGlobe(globeCanvas, (runId: string | null) => {
      this.selectRun(runId);
    });

    this.runList = new RunList(runListEl, (runId: string | null) => {
      this.selectRun(runId);
    });

    this.spillOverlay = new SpillOverlay(overlayEl);
    this.provenancePanel = new ProvenancePanel(provenanceEl);
    this.regionChips = new RegionChips(regionChipsEl, (regionId) => {
      void this.onRegionSelected(regionId);
    });
    this.modelInfoPanel = new ModelInfoPanel(this.modelInfoPanelEl);
    this.vesselDrawer = new VesselDrawer(drawerEl);

    this.init();
  }

  private async init(): Promise<void> {
    this.setupClearButton();
    this.setupRunListToggle();
    this.setupResponsive();
    this.setupRefreshButton();
    this.setupErrorButtons();
    this.setupReportButton();
    this.setupModelInfoToggle();
    this.setupZoomControls();
    void this.loadRegions();
    void this.loadModelInfo();
    await this.loadAllRuns();
  }

  private async loadRegions(): Promise<void> {
    try {
      this.regions = await listRegions();
    } catch (err) {
      // Non-fatal: the app works fine with no region switcher at all.
      console.warn('[App] failed to load regions:', err);
      this.regions = [];
    }
    this.regionChips.render(this.regions, this.selectedRegionId);
  }

  private async loadModelInfo(): Promise<void> {
    try {
      const info = await getModelInfo();
      this.modelInfoPanel.render(info);
    } catch (err) {
      // Non-fatal: the toggle just opens onto a loading state forever,
      // which is honest given the request itself failed.
      console.warn('[App] failed to load model info:', err);
    }
  }

  private setupModelInfoToggle(): void {
    this.modelInfoToggle.addEventListener('click', () => {
      this.modelInfoPanelEl.classList.toggle('translate-x-full');
    });
  }

  private async onRegionSelected(regionId: string | null): Promise<void> {
    this.selectedRegionId = regionId;
    this.regionChips.render(this.regions, this.selectedRegionId);
    await this.loadAllRuns();
  }

  private setState(state: 'loading' | 'error' | 'ready', errorMsg?: string): void {
    this.loadingEl.style.display = state === 'loading' ? 'flex' : 'none';
    this.errorEl.style.display = state === 'error' ? 'flex' : 'none';
    if (state === 'error' && errorMsg) {
      this.errorEl.querySelector('.error-message')!.textContent = errorMsg;
    }
  }

  private async loadAllRuns(): Promise<void> {
    this.setState('loading');

    try {
      const apiRuns = await listRuns(this.selectedRegionId);
      if (apiRuns.length > 0) {
        const full = await Promise.all(apiRuns.map((r) => getRun(r.scene_id)));
        this.allRuns = full;
      } else {
        this.allRuns = [];
      }
    } catch (err) {
      if (USE_MOCKS) {
        // Fixture fallback only when VITE_USE_MOCKS=true
        console.warn('[App] API unreachable, loading fixtures:', err);
        await this.loadFixtures();
      } else {
        const msg = err instanceof Error ? err.message : String(err);
        this.setState('error', msg);
        return;
      }
    }

    this.setState('ready');
    this.globe.init(this.allRuns);
    this.runList.render(this.allRuns);
    this.updateStatsStrip();
  }

  private async loadFixtures(): Promise<void> {
    const fixtureNames = ['run-with-vessels', 'run-with-drift', 'run-no-alert'];
    const loaded = await Promise.all(
      fixtureNames.map(async (name) => {
        try {
          const res = await fetch(`/src/mocks/fixtures/${name}.json`);
          return (await res.json()) as PipelineRun;
        } catch {
          return null;
        }
      }),
    );
    this.allRuns = loaded.filter((r): r is PipelineRun => r !== null);
  }

  private selectRun(runId: string | null): void {
    this.globe.selectSpill(runId);
    this.runList.setSelected(runId);

    if (runId === null) {
      this.spillOverlay.render(null);
      this.provenancePanel.render(null);
      this.vesselDrawer.hide();
      this.detailStack.classList.remove('drawer-open');
      this.globeHint.style.opacity = '1';
      this.selectedSceneId = null;
      this.reportBtn.hidden = true;
      return;
    }

    const run = this.allRuns.find(
      (r) => (r.alert?.spill_id || r.spill.scene_id) === runId,
    );
    if (!run) return;

    this.spillOverlay.render(run);
    this.provenancePanel.render(run);
    this.vesselDrawer.render(run.vessels);
    // Vessel drawer only actually opens when there's something to show
    // (VesselDrawer.render() no-ops to hidden on an empty list) — keep the
    // detail stack lifted only in that case, so it doesn't leave a dead gap
    // when there's no drawer to clear.
    this.detailStack.classList.toggle('drawer-open', run.vessels.length > 0);
    this.globeHint.style.opacity = '0';
    this.selectedSceneId = run.spill.scene_id;
    this.reportBtn.hidden = false;

    // On mobile, close the run list panel after selection
    if (window.innerWidth < 1024) {
      this.runListPanel.classList.add('-translate-x-full');
    }
  }

  private setupClearButton(): void {
    const btn = document.getElementById('clear-btn');
    if (!btn) return;
    btn.addEventListener('click', () => {
      this.selectRun(null);
    });
  }

  private setupRefreshButton(): void {
    const btn = document.getElementById('refresh-btn');
    if (!btn) return;
    btn.addEventListener('click', async () => {
      btn.classList.add('animate-spin');
      await this.loadAllRuns();
      btn.classList.remove('animate-spin');
    });
  }

  private setupErrorButtons(): void {
    const retryBtn = document.getElementById('error-retry-btn');
    const mockBtn = document.getElementById('error-mock-btn');
    retryBtn?.addEventListener('click', () => this.loadAllRuns());
    mockBtn?.addEventListener('click', async () => {
      await this.loadFixtures();
      this.setState('ready');
      this.globe.init(this.allRuns);
      this.runList.render(this.allRuns);
      this.updateStatsStrip();
    });
  }

  private updateStatsStrip(): void {
    const stripTotal = document.getElementById('strip-total');
    const stripActive = document.getElementById('strip-active');
    const stripVessels = document.getElementById('strip-vessels');
    if (stripTotal) stripTotal.textContent = String(this.allRuns.length);
    if (stripActive) stripActive.textContent = String(this.allRuns.filter(r => r.alert?.status && r.alert.status !== 'none').length);
    if (stripVessels) stripVessels.textContent = String(this.allRuns.reduce((s, r) => s + r.vessels.length, 0));
  }

  private setupReportButton(): void {
    this.reportBtn.addEventListener('click', async () => {
      if (!this.selectedSceneId) return;
      try {
        const blob = await getReport(this.selectedSceneId);
        const url = URL.createObjectURL(blob);
        window.open(url, '_blank');
        setTimeout(() => URL.revokeObjectURL(url), 60_000);
      } catch (err) {
        console.error('Failed to fetch report:', err);
      }
    });
  }

  private setupZoomControls(): void {
    document.getElementById('zoom-in-btn')?.addEventListener('click', () => this.globe.zoomIn());
    document.getElementById('zoom-out-btn')?.addEventListener('click', () => this.globe.zoomOut());
    document.getElementById('zoom-reset-btn')?.addEventListener('click', () => this.globe.resetZoom());
  }

  private setupRunListToggle(): void {
    this.runListToggle.addEventListener('click', () => {
      this.runListPanel.classList.toggle('-translate-x-full');
    });
  }

  private setupResponsive(): void {
    const mq = window.matchMedia('(max-width: 1023px)');

    const applyBreakpoint = (matches: boolean) => {
      if (matches) {
        this.runListToggle.classList.remove('hidden');
        this.runListPanel.classList.add('-translate-x-full');
      } else {
        this.runListToggle.classList.add('hidden');
        this.runListPanel.classList.remove('-translate-x-full');
      }
    };

    applyBreakpoint(mq.matches);
    mq.addEventListener('change', (e) => applyBreakpoint(e.matches));
  }

  destroy(): void {
    this.globe.destroy();
    this.runList.destroy();
    this.spillOverlay.destroy();
    this.provenancePanel.destroy();
    this.regionChips.destroy();
    this.modelInfoPanel.destroy();
    this.vesselDrawer.destroy();
  }
}
