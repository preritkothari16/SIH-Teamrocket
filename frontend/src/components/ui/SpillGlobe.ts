import createGlobe, { type Globe, type Marker } from 'cobe';
import type { PipelineRun } from '../../types/schema';

type StatusColor = [number, number, number];

const STATUS_COLORS: Record<string, StatusColor> = {
  new: [1, 0.3, 0.3],
  update: [1, 0.55, 0.15],
  possible: [0.95, 0.85, 0.2],
  none: [0.4, 0.4, 0.4],
};

const STATUS_SIZE: Record<string, number> = {
  new: 0.06,
  update: 0.05,
  possible: 0.04,
  none: 0.03,
};

interface ValidatedMarker {
  spillId: string;
  runIndex: number;
  lat: number;
  lon: number;
  cobeLocation: [number, number];
  confidence: number;
  status: 'new' | 'update' | 'possible' | 'none';
}

function isValidCoord(lat: unknown, lon: unknown): boolean {
  return (
    typeof lat === 'number' && typeof lon === 'number' &&
    isFinite(lat) && isFinite(lon) &&
    lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180
  );
}

function buildMarkers(runs: PipelineRun[]): ValidatedMarker[] {
  const markers: ValidatedMarker[] = [];
  for (let i = 0; i < runs.length; i++) {
    const run = runs[i];
    const lat = run.spill.centroid.lat;
    const lon = run.spill.centroid.lon;
    if (!isValidCoord(lat, lon)) {
      console.warn(`[SpillGlobe] skipping marker for ${run.spill.scene_id}: invalid centroid (${lat}, ${lon})`);
      continue;
    }
    markers.push({
      spillId: run.alert?.spill_id ?? run.spill.scene_id,
      runIndex: i,
      lat,
      lon,
      cobeLocation: [lat, lon],
      confidence: run.spill.confidence,
      status: run.alert?.status ?? 'none',
    });
  }
  return markers;
}

function toCobeMarkers(markers: ValidatedMarker[]): Marker[] {
  return markers.map((m) => ({
    location: m.cobeLocation,
    size: STATUS_SIZE[m.status] ?? 0.03,
    color: STATUS_COLORS[m.status] ?? STATUS_COLORS.none,
    id: m.spillId,
  }));
}

/** Convert a lat/lon to the phi/theta that centers that point on screen. */
function latLonToPhiTheta(latDeg: number, lonDeg: number): { phi: number; theta: number } {
  const phi = (lonDeg * Math.PI) / 180;
  let theta = (latDeg * Math.PI) / 180;
  theta = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, theta));
  return { phi, theta };
}

/** Shortest angular distance accounting for wraparound. */
function angleDist(a: number, b: number): number {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

/** Ease-out cubic. */
function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

export class SpillGlobe {
  private canvas: HTMLCanvasElement;
  private container: HTMLElement;
  private globe: Globe | null = null;
  private width = 0;
  private height = 0;
  private animFrame: number | null = null;
  private running = false;

  // Current rotation state
  private phi = 0;
  private theta = 0.3;

  // Idle auto-rotation
  private idlePhi = 0;
  private idleSpeed = 0.003;

  // Focus animation
  private focusAnimating = false;
  private focusStartTime = 0;
  private focusDuration = 1000;
  private focusFromPhi = 0;
  private focusFromTheta = 0;
  private focusToPhi = 0;
  private focusToTheta = 0;

  // Pointer / drag
  private dragging = false;
  private lastPointerX = 0;
  private lastPointerY = 0;
  private dragStartedAt: { x: number; y: number } | null = null;

  // Selection state
  private markers: ValidatedMarker[] = [];
  private selectedRunId: string | null = null;
  private onSelectSpill: (runId: string | null) => void;

  // Zoom state
  private zoomScale = 1;
  private zoomTarget = 1;

  constructor(canvas: HTMLCanvasElement, onSelectSpill: (runId: string | null) => void) {
    this.canvas = canvas;
    this.container = canvas.parentElement!;
    this.onSelectSpill = onSelectSpill;
  }

  init(runs: PipelineRun[]): void {
    this.destroy();
    this.markers = buildMarkers(runs);
    this.updateSize();

    this.globe = createGlobe(this.canvas, {
      devicePixelRatio: window.devicePixelRatio || 2,
      width: this.width,
      height: this.height,
      phi: this.phi,
      theta: this.theta,
      dark: 1,
      diffuse: 1.2,
      mapSamples: 16000,
      mapBrightness: 6,
      mapBaseBrightness: 0.05,
      baseColor: [0.1, 0.1, 0.15],
      markerColor: [1, 1, 1],
      glowColor: [0.12, 0.12, 0.2],
      markers: toCobeMarkers(this.markers),
      opacity: 1,
    });

    this.setupResize();
    this.setupPointer();
    this.start();
  }

  updateRuns(runs: PipelineRun[]): void {
    this.markers = buildMarkers(runs);
    this.globe?.update({ markers: toCobeMarkers(this.markers) });
  }

  selectSpill(runId: string | null): void {
    if (this.selectedRunId === runId) return;
    this.selectedRunId = runId;

    if (runId === null) {
      // Deselect — resume idle rotation, zoom out
      this.focusAnimating = false;
      this.zoomTarget = 1;
      return;
    }

    const marker = this.markers.find((m) => m.spillId === runId);
    if (!marker) return;

    const target = latLonToPhiTheta(marker.lat, marker.lon);
    // Animate phi taking the shortest path
    const phiDelta = angleDist(target.phi, this.phi);
    this.focusFromPhi = this.phi;
    this.focusToPhi = this.phi + phiDelta;
    this.focusFromTheta = this.theta;
    this.focusToTheta = target.theta;
    this.focusStartTime = performance.now();
    this.focusAnimating = true;

    // Fake zoom in
    this.zoomTarget = 1.4;
  }

  private updateSize(): void {
    const rect = this.container.getBoundingClientRect();
    if (rect) {
      this.width = rect.width;
      this.height = rect.height;
    }
  }

  private resizeObserver: ResizeObserver | null = null;

  private setupResize(): void {
    this.resizeObserver = new ResizeObserver(() => {
      this.updateSize();
      this.globe?.update({ width: this.width, height: this.height });
    });
    this.resizeObserver.observe(this.container);
  }

  private setupPointer(): void {
    const onPointerDown = (e: PointerEvent) => {
      this.dragging = true;
      this.lastPointerX = e.clientX;
      this.lastPointerY = e.clientY;
      this.dragStartedAt = { x: e.clientX, y: e.clientY };
      this.canvas.style.cursor = 'grabbing';

      // Cancel any in-flight focus animation — manual drag always wins
      if (this.focusAnimating) {
        this.focusAnimating = false;
        this.idlePhi = this.phi;
      }
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastPointerX;
      const dy = e.clientY - this.lastPointerY;
      this.phi += dx * 0.005;
      this.theta = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, this.theta + dy * 0.005));
      this.idlePhi = this.phi;
      this.lastPointerX = e.clientX;
      this.lastPointerY = e.clientY;
      this.globe?.update({ phi: this.phi, theta: this.theta });
    };

    const onPointerUp = (e: PointerEvent) => {
      if (!this.dragging) return;
      this.dragging = false;
      this.canvas.style.cursor = 'grab';

      if (this.dragStartedAt) {
        const dist = Math.hypot(
          e.clientX - this.dragStartedAt.x,
          e.clientY - this.dragStartedAt.y,
        );
        if (dist < 5) {
          this.handleClick(e);
        }
        this.dragStartedAt = null;
      }
    };

    this.canvas.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    this.canvas.style.cursor = 'grab';

    this._cleanupPointer = () => {
      this.canvas.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };
  }

  private _cleanupPointer: (() => void) | null = null;

  /**
   * Raycast from screen click to find the nearest visible marker.
   * Uses the same lat/lon → screen projection as marker placement.
   * For clustered markers (<1° apart), falls back to nearest-on-screen
   * since cobe doesn't expose per-marker hit testing.
   */
  private handleClick(e: PointerEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const nx = (x / rect.width) * 2 - 1;
    const ny = (y / rect.height) * 2 - 1;

    let bestDist = Infinity;
    let bestMarker: ValidatedMarker | null = null;

    for (const marker of this.markers) {
      const latRad = (marker.lat * Math.PI) / 180;
      const lonRad = (marker.lon * Math.PI) / 180;

      // Same projection as cobe's internal marker placement
      const gx = Math.cos(latRad) * Math.sin(lonRad - this.phi);
      const gy = Math.sin(latRad);
      const gz = Math.cos(latRad) * Math.cos(lonRad - this.phi);

      const ry = gy * Math.cos(this.theta) - gz * Math.sin(this.theta);
      const rz = gy * Math.sin(this.theta) + gz * Math.cos(this.theta);

      // Behind the globe — skip
      if (rz < 0) continue;

      const px = gx;
      const py = -ry;
      const dist = Math.hypot(px - nx, py - ny);

      if (dist < bestDist) {
        bestDist = dist;
        bestMarker = marker;
      }
    }

    if (bestMarker && bestDist < 0.15) {
      // Toggle: if clicking the already-selected marker, deselect
      const newId = bestMarker.spillId === this.selectedRunId ? null : bestMarker.spillId;
      this.onSelectSpill(newId);
    }
  }

  private start(): void {
    if (this.running) return;
    this.running = true;

    const animate = (now: number) => {
      if (!this.running) return;

      // Focus animation
      if (this.focusAnimating) {
        const elapsed = now - this.focusStartTime;
        const t = Math.min(elapsed / this.focusDuration, 1);
        const ease = easeOutCubic(t);
        this.phi = this.focusFromPhi + (this.focusToPhi - this.focusFromPhi) * ease;
        this.theta = this.focusFromTheta + (this.focusToTheta - this.focusFromTheta) * ease;
        this.idlePhi = this.phi;

        if (t >= 1) {
          this.focusAnimating = false;
        }
      } else if (!this.dragging) {
        // Idle auto-rotation (only when nothing is selected)
        if (this.selectedRunId === null) {
          this.idlePhi += this.idleSpeed;
          this.phi = this.idlePhi;
        }
      }

      // Zoom interpolation
      this.zoomScale += (this.zoomTarget - this.zoomScale) * 0.08;
      this.container.style.transform = `scale(${this.zoomScale})`;

      this.globe?.update({ phi: this.phi, theta: this.theta });
      this.animFrame = requestAnimationFrame(animate);
    };
    this.animFrame = requestAnimationFrame(animate);
  }

  destroy(): void {
    this.running = false;
    if (this.animFrame !== null) {
      cancelAnimationFrame(this.animFrame);
      this.animFrame = null;
    }
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    this._cleanupPointer?.();
    this._cleanupPointer = null;
    this.globe?.destroy();
    this.globe = null;
  }
}
