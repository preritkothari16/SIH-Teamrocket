import createGlobe, { type Globe } from 'cobe';
import type { PipelineRun } from '../../types/schema';

type RGB = [number, number, number];
export type Status = 'new' | 'update' | 'possible' | 'none';

const STATUS_COLORS: Record<Status, RGB> = {
  new:      [1.0, 0.15, 0.15],
  update:   [1.0, 0.55, 0.0],
  possible: [1.0, 0.85, 0.0],
  none:     [0.55, 0.57, 0.63],
};

/** Higher = more urgent. Picks a cluster's dominant (most urgent) status. */
const STATUS_SEVERITY: Record<Status, number> = { new: 3, update: 2, possible: 1, none: 0 };

/** Zoom is cobe's own native `scale` (crisp WebGL re-render), not a CSS
 *  transform — the previous CSS-scale approach blurred the canvas on
 *  zoom-in instead of actually re-rendering at the new size. */
const ZOOM_MIN = 0.6;
/** Capped mainly by cobe's own fixed dot-density base map (mapSamples
 *  below) starting to look sparse/blocky well past this, not by anything
 *  in the clustering math — markers themselves stay a fixed pixel size at
 *  any zoom, so there's no "oversized marker" risk from going higher. */
const ZOOM_MAX = 8;
const ZOOM_STEP_FACTOR = 1.4;
/** "Zoom moderately" for a single isolated spill — see spec section 7. */
const ZOOM_SINGLE_SELECT = 2.0;

/** Minimum on-screen pixel gap clustering tries to keep between two
 *  markers before folding them into one bubble. */
const CLUSTER_MIN_SEPARATION_PX = 28;
/** Cluster membership is recomputed only when the *target* zoom moves by
 *  more than this (relative, log-scale) since the last computation — not
 *  every animation frame — so panning/idle-rotation/in-flight zoom
 *  animation never re-run the clustering pass. See maybeRebuildClusters(). */
const CLUSTER_RECOMPUTE_LOG_THRESHOLD = 0.08;

export interface ValidatedMarker {
  spillId: string;
  runIndex: number;
  lat: number;
  lon: number;
  confidence: number;
  status: Status;
  areaKm2: number;
}

export interface ClusterGroup {
  id: string;
  lat: number;
  lon: number;
  members: ValidatedMarker[];
  dominantStatus: Status;
  /** Max great-circle distance (degrees) from centroid to any member —
   *  how "tight" this cluster is, used to size the zoom-in target. */
  spreadDeg: number;
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
      confidence: run.spill.confidence,
      status: run.alert?.status ?? 'none',
      areaKm2: run.spill.area_km2,
    });
  }
  return markers;
}

/** Fixed pixel radius, independent of zoom — the fix for "oversized
 *  markers": zooming in is what spreads overlapping points apart
 *  (clustering threshold shrinks as the globe's angular-to-pixel scale
 *  grows), not what makes each individual dot bigger. */
function markerDotRadius(m: ValidatedMarker): number {
  const base: Record<Status, number> = { new: 6, update: 5.5, possible: 5, none: 4 };
  const b = base[m.status] ?? 4;
  const confScale = 0.8 + m.confidence * 0.4; // 0.8x - 1.2x
  return b * confScale;
}

function clusterBubbleRadius(count: number): number {
  return Math.min(26, 12 + Math.sqrt(count) * 4);
}

/* ── Clustering (screen-space grouping only — real lat/lon never touched) ── */

export function angularDistanceDeg(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const toRad = Math.PI / 180;
  const p1 = lat1 * toRad;
  const p2 = lat2 * toRad;
  const dLon = (lon2 - lon1) * toRad;
  let cosD = Math.sin(p1) * Math.sin(p2) + Math.cos(p1) * Math.cos(p2) * Math.cos(dLon);
  cosD = Math.max(-1, Math.min(1, cosD)); // guard fp drift outside acos's domain
  return (Math.acos(cosD) * 180) / Math.PI;
}

/** How close two markers must be (in degrees) to fold into one bubble at
 *  a given zoom — shrinks as zoom grows, so the same dataset naturally
 *  goes world groups -> regional groups -> individual spills purely from
 *  the zoom level, no hardcoded region tiers. */
export function clusterThresholdDeg(zoom: number, w: number, h: number): number {
  const projR = (Math.min(w, h) / 2) * 0.9 * zoom;
  const thresholdRad = CLUSTER_MIN_SEPARATION_PX / Math.max(projR, 1);
  return (thresholdRad * 180) / Math.PI;
}

/** Grid-bucket grouping — O(n), same core approach real point-clustering
 *  libraries (e.g. Supercluster) use. An earlier greedy nearest-centroid
 *  version chained through data like "five points each ~0.3-0.5° from a
 *  neighbor, spanning 0.6° overall": each new point compared against the
 *  *running mean*, which kept drifting to stay just barely in range, so
 *  the whole set never split even near max zoom. Bucketing by a fixed
 *  grid cell (sized to the current threshold) has no such chaining path —
 *  a point is in whichever cell its own raw lat/lon falls into, full
 *  stop. Trade-off: two points a hair's width apart but on opposite sides
 *  of a cell boundary won't merge — a standard, minor, well-understood
 *  characteristic of grid clustering, not a bug. */
export function buildClusters(markers: ValidatedMarker[], thresholdDeg: number): ClusterGroup[] {
  const cellSize = Math.max(thresholdDeg, 1e-6);
  const buckets = new Map<string, ValidatedMarker[]>();

  for (const m of markers) {
    const key = `${Math.floor(m.lat / cellSize)}:${Math.floor(m.lon / cellSize)}`;
    const existing = buckets.get(key);
    if (existing) existing.push(m);
    else buckets.set(key, [m]);
  }

  return [...buckets.values()].map((members, idx) => {
    const lat = members.reduce((s, m) => s + m.lat, 0) / members.length;
    const lon = members.reduce((s, m) => s + m.lon, 0) / members.length;
    const spreadDeg = members.reduce(
      (max, m) => Math.max(max, angularDistanceDeg(m.lat, m.lon, lat, lon)),
      0,
    );
    const dominantStatus = members.reduce<Status>(
      (best, m) => (STATUS_SEVERITY[m.status] > STATUS_SEVERITY[best] ? m.status : best),
      'none',
    );
    return {
      id: members.length === 1 ? members[0].spillId : `cluster-${idx}-${members.length}`,
      lat, lon, members, dominantStatus, spreadDeg,
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

/* ── Overlay canvas — this now draws every marker/cluster visual. cobe's
 * own native marker rendering is unused (no `markers` passed to it): cobe
 * can't draw a numbered cluster bubble, so once that has to be hand-drawn
 * anyway, drawing singleton dots here too avoids double-rendering the same
 * point (cobe's own dot underneath *and* a decorated overlay on top) —
 * exactly the layering that made the old globe look cluttered. ────────── */

function projectToScreen(
  lat: number, lon: number,
  phi: number, theta: number,
  w: number, h: number,
  zoom: number,
): { x: number; y: number; z: number } | null {
  const latR = (lat * Math.PI) / 180;
  const lonR = (lon * Math.PI) / 180;

  const gx = Math.cos(latR) * Math.sin(lonR - phi);
  const gy = Math.sin(latR);
  const gz = Math.cos(latR) * Math.cos(lonR - phi);

  const ry = gy * Math.cos(theta) - gz * Math.sin(theta);
  const rz = gy * Math.sin(theta) + gz * Math.cos(theta);

  if (rz < 0) return null; // behind the globe

  const cx = w / 2;
  const cy = h / 2;
  const r = (Math.min(w, h) / 2) * 0.9 * zoom;

  return { x: cx + gx * r, y: cy - ry * r, z: rz };
}

function drawOverlay(
  ctx: CanvasRenderingContext2D,
  clusters: ClusterGroup[],
  phi: number, theta: number,
  w: number, h: number,
  zoom: number,
  selectedId: string | null,
  hoveredId: string | null,
): void {
  ctx.clearRect(0, 0, w, h);

  const projected = clusters
    .map((c) => {
      const p = projectToScreen(c.lat, c.lon, phi, theta, w, h, zoom);
      return p ? { ...c, screen: p } : null;
    })
    .filter((c): c is ClusterGroup & { screen: { x: number; y: number; z: number } } => c !== null);

  projected.sort((a, b) => a.screen.z - b.screen.z);

  for (const c of projected) {
    const { x, y, z } = c.screen;
    const alpha = 0.4 + z * 0.6;
    const rgb = STATUS_COLORS[c.dominantStatus] ?? STATUS_COLORS.none;
    const rgbCss = `${Math.round(rgb[0] * 255)},${Math.round(rgb[1] * 255)},${Math.round(rgb[2] * 255)}`;

    if (c.members.length === 1) {
      const member = c.members[0];
      const isSelected = member.spillId === selectedId;
      const isHovered = member.spillId === hoveredId && !isSelected;
      const dotR = markerDotRadius(member);

      ctx.beginPath();
      ctx.arc(x, y, dotR, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${rgbCss},${(0.95 * alpha).toFixed(3)})`;
      ctx.fill();

      // One subtle ring for state — never both at once, never stacked.
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(x, y, dotR + 5, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(255,255,255,${(0.85 * alpha).toFixed(3)})`;
        ctx.lineWidth = 2;
        ctx.stroke();
      } else if (isHovered) {
        ctx.beginPath();
        ctx.arc(x, y, dotR + 4, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(255,255,255,${(0.5 * alpha).toFixed(3)})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    } else {
      const isHovered = c.id === hoveredId;
      const bubbleR = clusterBubbleRadius(c.members.length);

      ctx.beginPath();
      ctx.arc(x, y, bubbleR, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${rgbCss},${(0.5 * alpha).toFixed(3)})`;
      ctx.fill();
      ctx.lineWidth = isHovered ? 2 : 1.5;
      ctx.strokeStyle = `rgba(${rgbCss},${(0.9 * alpha).toFixed(3)})`;
      ctx.stroke();

      ctx.font = `700 ${Math.min(13, 10 + bubbleR * 0.1)}px Inter, system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = `rgba(255,255,255,${alpha.toFixed(3)})`;
      ctx.fillText(String(c.members.length), x, y + 0.5);
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

  /** Two simultaneous pointers = pinch-zoom instead of drag-rotate. */
  private activePointers = new Map<number, { x: number; y: number }>();
  private pinchStartDist = 0;
  private pinchStartZoom = 1;

  private markers: ValidatedMarker[] = [];
  private clusters: ClusterGroup[] = [];
  private lastClusterZoom = 1;

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
    this.canvas.style.touchAction = 'none'; // let us handle pinch ourselves
  }

  init(runs: PipelineRun[]): void {
    this.destroy();
    this.markers = buildMarkers(runs);
    this.updateSize();
    this.resizeOverlay();
    this.rebuildClusters();

    this.globe = createGlobe(this.canvas, {
      devicePixelRatio: window.devicePixelRatio || 2,
      width: this.width,
      height: this.height,
      phi: this.phi,
      theta: this.theta,
      scale: this.zoomScale,
      dark: 1,
      diffuse: 1.2,
      mapSamples: 16000,
      mapBrightness: 6,
      mapBaseBrightness: 0.05,
      baseColor: [0.08, 0.08, 0.12],
      markerColor: [1, 1, 1],
      glowColor: [0.15, 0.12, 0.25],
      // No markers passed — every marker/cluster is hand-drawn on the 2D
      // overlay canvas instead (see the module doc comment above).
      opacity: 1,
    });

    this.setupResize();
    this.setupPointer();
    this.setupWheel();
    this.start();
  }

  updateRuns(runs: PipelineRun[]): void {
    this.markers = buildMarkers(runs);
    this.rebuildClusters();
  }

  selectSpill(runId: string | null): void {
    if (this.selectedRunId === runId) return;
    this.selectedRunId = runId;

    if (runId === null) {
      this.focusAnimating = false;
      this.setZoomTarget(1);
      return;
    }

    const marker = this.markers.find((m) => m.spillId === runId);
    if (!marker) return;

    this.focusOn(marker.lat, marker.lon);
    // Don't zoom OUT if the user already drilled further in via a cluster.
    this.setZoomTarget(Math.max(this.zoomTarget, ZOOM_SINGLE_SELECT));
  }

  /** Public zoom controls — wired to the +/-/reset UI in index.html. */
  zoomIn(): void {
    this.setZoomTarget(this.zoomTarget * ZOOM_STEP_FACTOR);
  }

  zoomOut(): void {
    this.setZoomTarget(this.zoomTarget / ZOOM_STEP_FACTOR);
  }

  resetZoom(): void {
    this.setZoomTarget(1);
  }

  private setZoomTarget(z: number): void {
    this.zoomTarget = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, z));
  }

  private focusOn(lat: number, lon: number): void {
    const target = latLonToPhiTheta(lat, lon);
    const phiDelta = angleDist(target.phi, this.phi);
    this.focusFromPhi = this.phi;
    this.focusToPhi = this.phi + phiDelta;
    this.focusFromTheta = this.theta;
    this.focusToTheta = target.theta;
    this.focusStartTime = performance.now();
    this.focusAnimating = true;
  }

  /** Density-aware zoom target for a cluster click (spec section 7): a
   *  tighter or larger cluster needs more zoom to visually separate than
   *  a loose one, computed from the cluster's own real spread — nothing
   *  hardcoded per region. */
  private zoomTargetForCluster(cluster: ClusterGroup): number {
    const desiredPx = Math.min(280, 40 + cluster.members.length * 18);
    const spreadRad = Math.max((cluster.spreadDeg * Math.PI) / 180, 0.01);
    const baseR = (Math.min(this.width, this.height) / 2) * 0.9;
    const targetProjR = desiredPx / Math.sin(spreadRad);
    return Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, targetProjR / baseR));
  }

  private rebuildClusters(): void {
    const thresholdDeg = clusterThresholdDeg(this.zoomTarget, this.width, this.height);
    this.clusters = buildClusters(this.markers, thresholdDeg);
    this.lastClusterZoom = this.zoomTarget;
  }

  /** Cheap guard, checked once a frame — only rebuilds (the actual O(n²)-
   *  ish work) when the target zoom has moved enough to plausibly change
   *  any grouping. Panning, idle rotation, and the smooth animation toward
   *  an already-decided target never trigger a rebuild. */
  private maybeRebuildClusters(): void {
    if (Math.abs(Math.log(this.zoomTarget / this.lastClusterZoom)) > CLUSTER_RECOMPUTE_LOG_THRESHOLD) {
      this.rebuildClusters();
    }
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
      this.rebuildClusters(); // pixel->degree threshold depends on canvas size
    });
    this.resizeObserver.observe(this.container);
  }

  private setupWheel(): void {
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = Math.exp(-e.deltaY * 0.0015);
      this.setZoomTarget(this.zoomTarget * factor);
    };
    this.canvas.addEventListener('wheel', onWheel, { passive: false });
    this._cleanupWheel = () => this.canvas.removeEventListener('wheel', onWheel);
  }
  private _cleanupWheel: (() => void) | null = null;

  private setupPointer(): void {
    const onPointerDown = (e: PointerEvent) => {
      this.canvas.setPointerCapture?.(e.pointerId);
      this.activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

      if (this.activePointers.size === 2) {
        // Pinch begins — stop any single-pointer drag/rotate.
        this.dragging = false;
        this.dragStartedAt = null;
        const pts = [...this.activePointers.values()];
        this.pinchStartDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        this.pinchStartZoom = this.zoomTarget;
        return;
      }

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
      if (this.activePointers.has(e.pointerId)) {
        this.activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      }

      if (this.activePointers.size === 2) {
        const pts = [...this.activePointers.values()];
        const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        if (this.pinchStartDist > 0) {
          this.setZoomTarget(this.pinchStartZoom * (dist / this.pinchStartDist));
        }
        return;
      }

      if (this.dragging) {
        const dx = e.clientX - this.lastPointerX;
        const dy = e.clientY - this.lastPointerY;
        this.phi += dx * 0.005;
        this.theta = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, this.theta + dy * 0.005));
        this.idlePhi = this.phi;
        this.lastPointerX = e.clientX;
        this.lastPointerY = e.clientY;
        return;
      }

      // Hover feedback — same affordance the sidebar rows and region
      // chips already give on :hover; the globe previously gave none at
      // all until the moment of an actual click.
      const hit = this.clusterAtClientPoint(e.clientX, e.clientY);
      const singleton = hit && hit.members.length === 1 ? hit.members[0].spillId : null;
      const newHoveredId = hit ? (singleton ?? hit.id) : null;
      if (newHoveredId !== this.hoveredRunId) {
        this.hoveredRunId = newHoveredId;
      }
      this.canvas.style.cursor = newHoveredId ? 'pointer' : 'grab';
    };

    const onPointerUp = (e: PointerEvent) => {
      this.canvas.releasePointerCapture?.(e.pointerId);
      this.activePointers.delete(e.pointerId);

      if (this.activePointers.size < 2) {
        this.pinchStartDist = 0;
      }
      // A second finger lifting off a pinch shouldn't resume a drag using
      // stale coordinates — require a fresh pointerdown to drag again.
      if (this.activePointers.size >= 1) return;

      if (!this.dragging) return;
      this.dragging = false;
      this.canvas.style.cursor = 'grab';

      if (this.dragStartedAt) {
        const dist = Math.hypot(
          e.clientX - this.dragStartedAt.x,
          e.clientY - this.dragStartedAt.y,
        );
        if (dist < 5) {
          this.handleClick(e.clientX, e.clientY);
        }
        this.dragStartedAt = null;
      }
    };

    this.canvas.addEventListener('pointerdown', onPointerDown);
    this.canvas.addEventListener('pointerleave', onPointerLeave);
    this.canvas.addEventListener('pointermove', onPointerMove);
    this.canvas.addEventListener('pointerup', onPointerUp);
    this.canvas.addEventListener('pointercancel', onPointerUp);
    this.canvas.style.cursor = 'grab';

    this._cleanupPointer = () => {
      this.canvas.removeEventListener('pointerdown', onPointerDown);
      this.canvas.removeEventListener('pointerleave', onPointerLeave);
      this.canvas.removeEventListener('pointermove', onPointerMove);
      this.canvas.removeEventListener('pointerup', onPointerUp);
      this.canvas.removeEventListener('pointercancel', onPointerUp);
    };
  }

  private _cleanupPointer: (() => void) | null = null;

  /** Hit-test against the CURRENT cluster list, in real screen pixels via
   *  the exact same projection used to draw them — click, hover, and
   *  rendering can never disagree about where something is, because they
   *  all go through projectToScreen(). */
  private clusterAtClientPoint(clientX: number, clientY: number): ClusterGroup | null {
    const rect = this.canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;

    let best: ClusterGroup | null = null;
    let bestDist = Infinity;

    for (const c of this.clusters) {
      const p = projectToScreen(c.lat, c.lon, this.phi, this.theta, this.width, this.height, this.zoomScale);
      if (!p) continue;
      const hitR = (c.members.length === 1 ? markerDotRadius(c.members[0]) : clusterBubbleRadius(c.members.length)) + 8;
      const dist = Math.hypot(p.x - px, p.y - py);
      if (dist <= hitR && dist < bestDist) {
        bestDist = dist;
        best = c;
      }
    }

    return best;
  }

  private handleClick(clientX: number, clientY: number): void {
    const hit = this.clusterAtClientPoint(clientX, clientY);
    if (!hit) return;

    if (hit.members.length === 1) {
      const member = hit.members[0];
      const newId = member.spillId === this.selectedRunId ? null : member.spillId;
      // selectSpill() (called back via App.ts's onSelectSpill round-trip)
      // does the actual focus+zoom — don't duplicate it here.
      this.onSelectSpill(newId);
    } else {
      this.focusOnCluster(hit);
    }
  }

  /** Clicking a cluster never selects a run or opens any panel — it's
   *  pure navigation, same as clicking empty water to look around. */
  private focusOnCluster(cluster: ClusterGroup): void {
    this.focusOn(cluster.lat, cluster.lon);
    this.setZoomTarget(this.zoomTargetForCluster(cluster));
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
      } else if (!this.dragging && this.activePointers.size === 0) {
        if (this.selectedRunId === null) {
          this.idlePhi += this.idleSpeed;
          this.phi = this.idlePhi;
        }
      }

      this.zoomScale += (this.zoomTarget - this.zoomScale) * 0.08;
      this.maybeRebuildClusters();
      // Cheap, harmless verification hooks — not read by any app code.
      this.canvas.dataset.zoom = this.zoomScale.toFixed(3);
      this.canvas.dataset.clusterCount = String(this.clusters.length);

      this.globe?.update({ phi: this.phi, theta: this.theta, scale: this.zoomScale });

      drawOverlay(
        this.overlayCtx, this.clusters,
        this.phi, this.theta,
        this.width, this.height,
        this.zoomScale,
        this.selectedRunId, this.hoveredRunId,
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
    this._cleanupWheel?.();
    this._cleanupWheel = null;
    this.activePointers.clear();
    this.hoveredRunId = null;
    this.globe?.destroy();
    this.globe = null;
    this.overlayCtx.clearRect(0, 0, this.width, this.height);
  }
}
