import type { PipelineRun } from '../types/schema';

export interface RunSummary {
  scene_id: string;
  acquisition_timestamp: string | null;
  area_km2: number;
  confidence: number;
  alert_status: string;
}

export async function listRuns(): Promise<RunSummary[]> {
  const res = await fetch('/api/runs');
  if (!res.ok) throw new Error(`GET /api/runs failed: ${res.status}`);
  const data = await res.json();
  return data.value ?? data;
}

export async function getRun(sceneId: string): Promise<PipelineRun> {
  const res = await fetch(`/api/runs/${encodeURIComponent(sceneId)}`);
  if (!res.ok) throw new Error(`GET /api/runs/${sceneId} failed: ${res.status}`);
  return res.json();
}

export async function getReport(sceneId: string): Promise<Blob> {
  const res = await fetch(`/api/runs/${encodeURIComponent(sceneId)}/report`);
  if (!res.ok) throw new Error(`GET /api/runs/${sceneId}/report failed: ${res.status}`);
  return res.blob();
}

export async function triggerRun(scenePath: string): Promise<PipelineRun> {
  const res = await fetch('/api/runs', {
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
