import type { Alert, PipelineRun } from '../types/schema';

// Empty string keeps requests relative, routed through vite.config.ts's dev
// proxy to localhost:8000. Set VITE_API_BASE_URL (frontend/.env) to hit a
// backend directly — e.g. a prod build with no dev-server proxy.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';

export interface RunSummary {
  scene_id: string;
  acquisition_timestamp: string | null;
  area_km2: number;
  confidence: number;
  // Same vocabulary as PipelineRun.alert.status (src/api/registry.py's
  // shared _map_alert_status) — both endpoints run a spill's raw backend
  // status through the same mapping now.
  alert_status: Alert['status'];
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function listRuns(): Promise<RunSummary[]> {
  const res = await fetch(`${API_BASE_URL}/api/runs`);
  if (!res.ok) throw new Error(`GET /api/runs failed: ${res.status}`);
  const data = await res.json();
  return data.value ?? data;
}

export async function getRun(sceneId: string): Promise<PipelineRun> {
  const res = await fetch(`${API_BASE_URL}/api/runs/${encodeURIComponent(sceneId)}`);
  if (!res.ok) throw new Error(`GET /api/runs/${sceneId} failed: ${res.status}`);
  return res.json();
}

export async function getReport(sceneId: string): Promise<Blob> {
  const res = await fetch(`${API_BASE_URL}/api/runs/${encodeURIComponent(sceneId)}/report`);
  if (!res.ok) throw new Error(`GET /api/runs/${sceneId}/report failed: ${res.status}`);
  return res.blob();
}

export async function triggerRun(scenePath: string): Promise<PipelineRun> {
  const res = await fetch(`${API_BASE_URL}/api/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scene_path: scenePath, stub_model: true }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `POST /api/runs failed: ${res.status}`);
  }
  return res.json();
}
