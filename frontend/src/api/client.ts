import type { PipelineRun } from '../types/schema';

export interface RunSummary {
  scene_id: string;
  acquisition_timestamp: string | null;
  area_km2: number;
  confidence: number;
  alert_status: string;
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '');

// ---------------------------------------------------------------------------
// Lightweight runtime validation — catches drifted backend shapes without
// pulling in zod.  Returns the data if valid, throws a descriptive Error
// otherwise.
// ---------------------------------------------------------------------------
function validateRunSummary(data: unknown): RunSummary[] {
  if (!Array.isArray(data)) throw new Error('Expected array from GET /api/runs');
  return data.map((item, i) => {
    if (typeof item !== 'object' || item === null) throw new Error(`RunSummary[${i}] is not an object`);
    const obj = item as Record<string, unknown>;
    if (typeof obj.scene_id !== 'string') throw new Error(`RunSummary[${i}].scene_id is not a string`);
    if (typeof obj.area_km2 !== 'number') throw new Error(`RunSummary[${i}].area_km2 is not a number`);
    if (typeof obj.confidence !== 'number') throw new Error(`RunSummary[${i}].confidence is not a number`);
    return {
      scene_id: obj.scene_id,
      acquisition_timestamp: (typeof obj.acquisition_timestamp === 'string' ? obj.acquisition_timestamp : null),
      area_km2: obj.area_km2,
      confidence: obj.confidence,
      alert_status: typeof obj.alert_status === 'string' ? obj.alert_status : 'none',
    };
  });
}

function validatePipelineRun(data: unknown): PipelineRun {
  if (typeof data !== 'object' || data === null) throw new Error('Expected object from GET /api/runs/:id');
  const obj = data as Record<string, unknown>;

  if (typeof obj.spill !== 'object' || obj.spill === null) throw new Error('PipelineRun.spill is missing');
  const spill = obj.spill as Record<string, unknown>;
  if (typeof spill.scene_id !== 'string') throw new Error('PipelineRun.spill.scene_id is not a string');
  if (typeof spill.centroid !== 'object' || spill.centroid === null) throw new Error('PipelineRun.spill.centroid is missing');

  if (typeof obj.alert !== 'object' || obj.alert === null) throw new Error('PipelineRun.alert is missing');

  if (!Array.isArray(obj.vessels)) throw new Error('PipelineRun.vessels is not an array');
  if (typeof obj.drift !== 'object' || obj.drift === null) throw new Error('PipelineRun.drift is missing');

  return data as PipelineRun;
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------
export async function listRuns(): Promise<RunSummary[]> {
  const res = await fetch(`${API_BASE}/api/runs`);
  if (!res.ok) throw new Error(`GET /api/runs failed: ${res.status} ${res.statusText}`);
  const data = await res.json();
  return validateRunSummary(data.value ?? data);
}

export async function getRun(sceneId: string): Promise<PipelineRun> {
  const res = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(sceneId)}`);
  if (!res.ok) throw new Error(`GET /api/runs/${sceneId} failed: ${res.status} ${res.statusText}`);
  const data = await res.json();
  return validatePipelineRun(data);
}

export async function getReport(sceneId: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(sceneId)}/report`);
  if (!res.ok) throw new Error(`GET /api/runs/${sceneId}/report failed: ${res.status} ${res.statusText}`);
  return res.blob();
}

export async function triggerRun(scenePath: string): Promise<PipelineRun> {
  const res = await fetch(`${API_BASE}/api/runs`, {
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
