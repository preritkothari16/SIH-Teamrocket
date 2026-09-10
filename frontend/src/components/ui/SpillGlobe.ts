import createGlobe, { type Globe, type Marker } from 'cobe';
import type { PipelineRun } from '../../types/schema';

type RGB = [number, number, number];

/** Vivid, high-contrast colors — brighter than before for dark globe. */
const STATUS_COLORS: Record<string, RGB> = {
  new:     [1.0, 0.2, 0.2],   // bright red
  update:  [1.0, 0.6, 0.0],   // vivid orange
  possible:[1.0, 0.9, 0.1],   // bright yellow
  none:    [0.45, 0.45, 0.5], // muted grey
};

/** Glow halos (slightly desaturated, drawn on overlay). */
const STATUS_GLOW: Record<string, string> = {
  new:      'rgba(255,50,50,',
  update:   'rgba(255,150,0,',
  possible: 'rgba(255,230,30,',
  none:     'rgba(120,120,130,',
};

const STATUS_LABELS: Record<string, string> = {
  new: 'NEW',
  update: 'UPD',
  possible: 'POSS',
  none: '',
};

interface ValidatedMarker {
  spillId: string;
  runIndex: number;
  lat: number;
  lon: number;
  cobeLocation: [number, number];
  confidence: number;
  status: 'new' | 'update' | 'possible' | 'none';
  areaKm2: number;
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
      areaKm2: run.spill.area_km2,
    });
  }
  return markers;
}

/**
 * Marker size = base(status) × confidence scale.
 * "new" markers are 2× the size of "none" markers.
 * Confidence scales between 0.7× and 1.3×.
 */
function markerSize(m: ValidatedMarker): number {
  const base: Record<string, number> = {
    new: 0.08,
    update: 0.065,
    possible: 0.05,
    none: 0.035,
  };
  const b = base[m.status] ?? 0.035;
  const confScale = 0.7 + m.confidence * 0.6;
  return b * confScale;
}

function toCobeMarkers(markers: ValidatedMarker[], selectedId: string | null): Marker[] {
  return markers.map((m) => {
    const isSelected = m.spillId === selectedId;
    const size = isSelected ? markerSize(m) * 1.5 : markerSize(m);
    return {
      location: m.cobeLocation,
      size,
      color: STATUS_COLORS[m.status] ?? STATUS_COLORS.none,
      id: m.spillId,
    };
  });
}

function latLonToPhiTheta(latDeg: number, lonDeg: number): { phi: number; theta: number } {
  const phi = (lonDeg * Math.PI) / 180;
  let theta = (latDeg * Math.PI) / 180;
  theta = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, theta));
  return { phi, theta };
}

function angleDist(a: number, b: number): number {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

/* ── Overlay canvas helpers ─────────────────────────────────────────── */

function projectToScreen(
  lat: number, lon: number,
  phi: number, theta: number,
  w: number, h: number,
): { x: number; y: number; z: number } | null {
  const latR = (lat * Math.PI) / 180;
  const lonR = (lon * Math.PI) / 180;

  const gx = Math.cos(latR) * Math.sin(lonR - phi);
  const gy = Math.sin(latR);
  const gz = Math.cos(latR) * Math.cos(lonR - phi);

  const ry = gy * Math.cos(theta) - gz * Math.sin(theta);
  const rz = gy * Math.sin(theta) + gz * Math.cos(theta);

  if (rz < 0) return null; // behind globe

  const cx = w / 2;
  const cy = h / 2;
  const r = Math.min(w, h) / 2 * 0.9;

  return { x: cx + gx * r, y: cy - ry * r, z: rz };
}

function drawOverlay(
  ctx: CanvasRenderingContext2D,
  markers: ValidatedMarker[],
  phi: number, theta: number,
  w: number, h: number,
  selectedId: string | null,
  now: number,
): void {
  ctx.clearRect(0, 0, w, h);

  const projected = markers
    .map((m) => {
      const p = projectToScreen(m.lat, m.lon, phi, theta, w, h);
      return p ? { ...m, screen: p } : null;
    })
    .filter(Boolean) as (ValidatedMarker & { screen: { x: number; y: number; z: number } })[];

  projected.sort((a, b) => a.screen.z - b.screen.z);

  for (const m of projected) {
    const { x, y, z } = m.screen;
    const isSelected = m.spillId === selectedId;
    const baseR = markerSize(m) * Math.min(w, h) / 2 * 0.9;
    const alpha = 0.35 + z * 0.65;

    const glowBase = STATUS_GLOW[m.status] ?? STATUS_GLOW.none;

    // ── Subtle glow halo (contained, not bleeding) ──
    const glowR = baseR * (isSelected ? 3 : 2.2);
    const grad = ctx.createRadialGradient(x, y, baseR * 0.3, x, y, glowR);
    grad.addColorStop(0, glowBase + (0.5 * alpha).toFixed(3) + ')');
    grad.addColorStop(0.6, glowBase + (0.15 * alpha).toFixed(3) + ')');
    grad.addColorStop(1, glowBase + '0)');
    ctx.beginPath();
    ctx.arc(x, y, glowR, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    // ── Center dot (always visible, anchors the marker) ──
    ctx.beginPath();
    ctx.arc(x, y, Math.max(3, baseR * 0.35), 0, Math.PI * 2);
    const c = STATUS_COLORS[m.status] ?? STATUS_COLORS.none;
    ctx.fillStyle = `rgba(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)},${(0.95 * alpha).toFixed(3)})`;
    ctx.fill();
    // white outline for contrast
    ctx.strokeStyle = `rgba(255,255,255,${(0.5 * alpha).toFixed(3)})`;
    ctx.lineWidth = 1;
    ctx.stroke();

    // ── Pulse ring for "new" status ──
    if (m.status === 'new') {
      const pulseT = ((now / 1400) + m.runIndex * 0.3) % 1;
      const pulseR = baseR * 1.5 + pulseT * baseR * 2.5;
      const pulseAlpha = (1 - pulseT) * 0.5 * alpha;
      ctx.beginPath();
      ctx.arc(x, y, pulseR, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,70,70,${pulseAlpha.toFixed(3)})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // ── Selection ring ──
    if (isSelected) {
      const ringR = baseR * 2;
      ctx.beginPath();
      ctx.arc(x, y, ringR, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,255,255,${(0.8 * alpha).toFixed(3)})`;
      ctx.lineWidth = 2.5;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(x, y, ringR + 4, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,255,255,${(0.2 * alpha).toFixed(3)})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // ── Status label (directly above marker) ──
    const label = STATUS_LABELS[m.status];
    if (label) {
      const fontSize = Math.max(9, Math.min(13, baseR * 0.9));
      ctx.font = `700 ${fontSize}px Inter, system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      const labelY = y - baseR * 1.4 - 4;
      // text shadow for contrast
      ctx.fillStyle = `rgba(0,0,0,${(0.7 * alpha).toFixed(3)})`;
      ctx.fillText(label, x + 1, labelY + 1);
      const labelAlpha = isSelected ? 1 : 0.75;
      ctx.fillStyle = `rgba(255,255,255,${(labelAlpha * alpha).toFixed(3)})`;
      ctx.fillText(label, x, labelY);
    }
  }
}

/* ── SpillGlobe class ──────────────────────────────────────────────── */

export class SpillGlobe {
  private canvas: HTMLCanvasElement;
  private overlayCanvas: HTMLCanvasElement;
  private overlayCtx: CanvasRenderingContext2D;
  private container: HTMLElement;
  private globe: Globe | null = null;
  private width = 0;
  private height = 0;
  private animFrame: number | null = null;
  private running = false;

  private phi = 0;
  private theta = 0.3;

  private idlePhi = 0;
  private idleSpeed = 0.003;

  private focusAnimating = false;
  private focusStartTime = 0;
  private focusDuration = 1000;
  private focusFromPhi = 0;
  private focusFromTheta = 0;
  private focusToPhi = 0;
  private focusToTheta = 0;

  private dragging = false;
  private lastPointerX = 0;
  private lastPointerY = 0;
  private dragStartedAt: { x: number; y: number } | null = null;

  private markers: ValidatedMarker[] = [];
  private selectedRunId: string | null = null;
  private onSelectSpill: (runId: string | null) => void;

  private zoomScale = 1;
  private zoomTarget = 1;

  constructor(canvas: HTMLCanvasElement, onSelectSpill: (runId: string | null) => void) {
    this.canvas = canvas;
    this.container = canvas.parentElement!;
    this.overlayCanvas = this.container.querySelector('#globe-overlay') as HTMLCanvasElement;
    this.overlayCtx = this.overlayCanvas.getContext('2d')!;
    this.onSelectSpill = onSelectSpill;
  }

  init(runs: PipelineRun[]): void {
    this.destroy();
    this.markers = buildMarkers(runs);
    this.updateSize();
    this.resizeOverlay();

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
      baseColor: [0.08, 0.08, 0.12],
      markerColor: [1, 1, 1],
      glowColor: [0.15, 0.12, 0.25],
      markers: toCobeMarkers(this.markers, null),
      opacity: 1,
    });

    this.setupResize();
    this.setupPointer();
    this.start();
  }

  updateRuns(runs: PipelineRun[]): void {
    this.markers = buildMarkers(runs);
    this.globe?.update({ markers: toCobeMarkers(this.markers, this.selectedRunId) });
  }

  selectSpill(runId: string | null): void {
    if (this.selectedRunId === runId) return;
    this.selectedRunId = runId;

    this.globe?.update({ markers: toCobeMarkers(this.markers, runId) });

    if (runId === null) {
      this.focusAnimating = false;
      this.zoomTarget = 1;
      return;
    }

    const marker = this.markers.find((m) => m.spillId === runId);
    if (!marker) return;

    const target = latLonToPhiTheta(marker.lat, marker.lon);
    const phiDelta = angleDist(target.phi, this.phi);
    this.focusFromPhi = this.phi;
    this.focusToPhi = this.phi + phiDelta;
    this.focusFromTheta = this.theta;
    this.focusToTheta = target.theta;
    this.focusStartTime = performance.now();
    this.focusAnimating = true;
    this.zoomTarget = 1.4;
  }

  private updateSize(): void {
    const rect = this.container.getBoundingClientRect();
    if (rect) {
      this.width = rect.width;
      this.height = rect.height;
    }
  }

  private resizeOverlay(): void {
    const dpr = window.devicePixelRatio || 2;
    this.overlayCanvas.width = this.width * dpr;
    this.overlayCanvas.height = this.height * dpr;
    this.overlayCanvas.style.width = this.width + 'px';
    this.overlayCanvas.style.height = this.height + 'px';
    this.overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  private resizeObserver: ResizeObserver | null = null;

  private setupResize(): void {
    this.resizeObserver = new ResizeObserver(() => {
      this.updateSize();
      this.resizeOverlay();
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

      const gx = Math.cos(latRad) * Math.sin(lonRad - this.phi);
      const gy = Math.sin(latRad);
      const gz = Math.cos(latRad) * Math.cos(lonRad - this.phi);

      const ry = gy * Math.cos(this.theta) - gz * Math.sin(this.theta);
      const rz = gy * Math.sin(this.theta) + gz * Math.cos(this.theta);

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
      const newId = bestMarker.spillId === this.selectedRunId ? null : bestMarker.spillId;
      this.onSelectSpill(newId);
    }
  }

  private start(): void {
    if (this.running) return;
    this.running = true;

    const animate = (now: number) => {
      if (!this.running) return;

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
        if (this.selectedRunId === null) {
          this.idlePhi += this.idleSpeed;
          this.phi = this.idlePhi;
        }
      }

      this.zoomScale += (this.zoomTarget - this.zoomScale) * 0.08;
      this.container.style.transform = `scale(${this.zoomScale})`;

      this.globe?.update({ phi: this.phi, theta: this.theta });

      // Draw overlay effects
      drawOverlay(
        this.overlayCtx, this.markers,
        this.phi, this.theta,
        this.width, this.height,
        this.selectedRunId, now,
      );

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
    this.overlayCtx.clearRect(0, 0, this.width, this.height);
  }
}
