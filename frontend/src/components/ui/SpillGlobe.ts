import createGlobe, { type Globe, type Marker } from 'cobe';
import type { PipelineRun } from '../../types/schema';

type RGB = [number, number, number];

/** Vivid, high-contrast colors — boosted saturation for dark globe. */
const STATUS_COLORS: Record<string, RGB> = {
  new:     [1.0, 0.15, 0.15],   // intense red
  update:  [1.0, 0.55, 0.0],    // vivid orange
  possible:[1.0, 0.85, 0.0],    // bright yellow
  none:    [0.4, 0.42, 0.48],   // muted grey
};

/** Glow halos — more saturated for bolder bloom. */
const STATUS_GLOW: Record<string, string> = {
  new:      'rgba(255,30,30,',
  update:   'rgba(255,130,0,',
  possible: 'rgba(255,210,0,',
  none:     'rgba(100,105,120,',
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
      spillId: run.alert?.spill_id || run.spill.scene_id,
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
 * "new" markers are 1.8× the size of "none" markers.
 * Confidence scales between 0.7× and 1.3×.
 */
function markerSize(m: ValidatedMarker): number {
  const base: Record<string, number> = {
    new: 0.12,
    update: 0.11,
    possible: 0.09,
    none: 0.07,
  };
  const b = base[m.status] ?? 0.05;
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
  hoveredId: string | null,
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
    const isHovered = m.spillId === hoveredId && !isSelected;
    const baseR = markerSize(m) * Math.min(w, h) / 2 * 0.9;
    const alpha = 0.35 + z * 0.65;

    const glowBase = STATUS_GLOW[m.status] ?? STATUS_GLOW.none;
    const isNew = m.status === 'new';

    // ── Glow halo — tighter, less aggressive ──
    const glowR = baseR * (isNew ? 2 : isSelected ? 3 : 2.2);
    const glowIntensity = isNew ? 0.5 : 0.4;
    const grad = ctx.createRadialGradient(x, y, baseR * 0.2, x, y, glowR);
    grad.addColorStop(0, glowBase + (glowIntensity * alpha).toFixed(3) + ')');
    grad.addColorStop(0.4, glowBase + ((glowIntensity * 0.4) * alpha).toFixed(3) + ')');
    grad.addColorStop(1, glowBase + '0)');
    ctx.beginPath();
    ctx.arc(x, y, glowR, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    // ── Reticle ring (outer thin ring, targeting feel) ──
    const c = STATUS_COLORS[m.status] ?? STATUS_COLORS.none;
    const ringR = Math.max(6, baseR * (isNew ? 1.1 : 0.9));
    ctx.beginPath();
    ctx.arc(x, y, ringR, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)},${(0.8 * alpha).toFixed(3)})`;
    ctx.lineWidth = isNew ? 2 : 1.5;
    ctx.stroke();

    // ── Inner dot (solid, anchors the marker) ──
    const dotR = Math.max(2.5, baseR * (isNew ? 0.28 : 0.2));
    ctx.beginPath();
    ctx.arc(x, y, dotR, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)},${(0.95 * alpha).toFixed(3)})`;
    ctx.fill();

    // ── Pulse ring for "new" status — smaller, tighter, synced ──
    if (isNew) {
      const pulseT = ((now / 700) + m.runIndex * 0.3) % 1;
      const pulseR = ringR * 1.1 + pulseT * ringR * 2;
      const pulseAlpha = (1 - pulseT) * 0.5 * alpha;
      ctx.beginPath();
      ctx.arc(x, y, pulseR, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,40,40,${pulseAlpha.toFixed(3)})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // ── Selection ring ──
    if (isSelected) {
      const selRingR = baseR * 2;
      ctx.beginPath();
      ctx.arc(x, y, selRingR, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,255,255,${(0.85 * alpha).toFixed(3)})`;
      ctx.lineWidth = 2.5;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(x, y, selRingR + 4, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,255,255,${(0.25 * alpha).toFixed(3)})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // ── Hover ring — lighter than selection, gives the same "this is
    // clickable" feedback the sidebar rows and region chips already have
    // on hover. Skipped when selected so the two rings don't compete. ──
    if (isHovered) {
      const hoverRingR = baseR * 1.6;
      ctx.beginPath();
      ctx.arc(x, y, hoverRingR, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,255,255,${(0.5 * alpha).toFixed(3)})`;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // ── Status label (directly above marker) ──
    const label = STATUS_LABELS[m.status];
    if (label) {
      const fontSize = Math.max(10, Math.min(14, baseR * 0.85));
      ctx.font = `700 ${fontSize}px Inter, system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      const labelY = y - baseR * 1.4 - 6;
      // text shadow for contrast
      ctx.fillStyle = `rgba(0,0,0,${(0.8 * alpha).toFixed(3)})`;
      ctx.fillText(label, x + 1, labelY + 1);
      const labelAlpha = isSelected ? 1 : 0.8;
      ctx.fillStyle = `rgba(255,255,255,${(labelAlpha * alpha).toFixed(3)})`;
      ctx.fillText(label, x, labelY);
    }
  }

  // ── Scan-line sweep (decorative rotating arc) ──
  // Only draw when no spill is selected/focused, to avoid visual clutter.
  // Also skip when reduced-motion is enabled.
  const isSelecting = selectedId !== null;
  const reducedMotion = document.body.dataset.reducedMotion === 'true';
  if (!isSelecting && !reducedMotion) {
    const scanAngle = (now / 6000) * Math.PI * 2; // 6s full rotation
    const cx = w / 2;
    const cy = h / 2;
    const scanR = Math.min(w, h) / 2 * 0.85;
    ctx.save();
    ctx.globalAlpha = 0.12;
    ctx.strokeStyle = 'rgba(100,200,255,1)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, scanR, scanAngle, scanAngle + 0.6);
    ctx.stroke();
    // trailing fade
    ctx.globalAlpha = 0.04;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(cx, cy, scanR, scanAngle - 0.3, scanAngle);
    ctx.stroke();
    ctx.restore();
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
  private hoveredRunId: string | null = null;
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
      this.hoveredRunId = null;

      if (this.focusAnimating) {
        this.focusAnimating = false;
        this.idlePhi = this.phi;
      }
    };

    const onPointerLeave = () => {
      this.hoveredRunId = null;
      if (!this.dragging) this.canvas.style.cursor = 'grab';
    };

    const onPointerMove = (e: PointerEvent) => {
      if (this.dragging) {
        const dx = e.clientX - this.lastPointerX;
        const dy = e.clientY - this.lastPointerY;
        this.phi += dx * 0.005;
        this.theta = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, this.theta + dy * 0.005));
        this.idlePhi = this.phi;
        this.lastPointerX = e.clientX;
        this.lastPointerY = e.clientY;
        this.globe?.update({ phi: this.phi, theta: this.theta });
        return;
      }

      // Hover feedback — same "this is clickable" affordance the sidebar
      // rows and region chips already give on :hover; the globe previously
      // gave none at all until the moment of an actual click.
      const marker = this.markerAtClientPoint(e.clientX, e.clientY);
      const newHoveredId = marker?.spillId ?? null;
      if (newHoveredId !== this.hoveredRunId) {
        this.hoveredRunId = newHoveredId;
      }
      this.canvas.style.cursor = newHoveredId ? 'pointer' : 'grab';
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
    this.canvas.addEventListener('pointerleave', onPointerLeave);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    this.canvas.style.cursor = 'grab';

    this._cleanupPointer = () => {
      this.canvas.removeEventListener('pointerdown', onPointerDown);
      this.canvas.removeEventListener('pointerleave', onPointerLeave);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };
  }

  private _cleanupPointer: (() => void) | null = null;

  /** Hit-test in the globe's own unrotated-sphere space (gx/gy/rz — see
   *  projectToScreen's own math) against every visible-face marker,
   *  nearest wins. Shared by the click handler and pointer-move hover
   *  detection so the two can never disagree about what's under the
   *  cursor — before this they'd have been two separate copies of the
   *  same math to keep in sync by hand. */
  private findMarkerAtNDC(nx: number, ny: number): ValidatedMarker | null {
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

      if (rz < 0) continue; // behind the globe, not visible

      const px = gx;
      const py = -ry;
      const dist = Math.hypot(px - nx, py - ny);

      if (dist < bestDist) {
        bestDist = dist;
        bestMarker = marker;
      }
    }

    return bestMarker && bestDist < 0.15 ? bestMarker : null;
  }

  private markerAtClientPoint(clientX: number, clientY: number): ValidatedMarker | null {
    const rect = this.canvas.getBoundingClientRect();
    const nx = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ny = ((clientY - rect.top) / rect.height) * 2 - 1;
    return this.findMarkerAtNDC(nx, ny);
  }

  private handleClick(e: PointerEvent): void {
    const marker = this.markerAtClientPoint(e.clientX, e.clientY);
    if (marker) {
      const newId = marker.spillId === this.selectedRunId ? null : marker.spillId;
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
        this.selectedRunId, this.hoveredRunId, now,
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
    this.hoveredRunId = null;
    this.globe?.destroy();
    this.globe = null;
    this.overlayCtx.clearRect(0, 0, this.width, this.height);
  }
}
