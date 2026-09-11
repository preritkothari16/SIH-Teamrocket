export interface Point {
  lat: number
  lon: number
}

export interface BBox {
  minLon: number
  minLat: number
  maxLon: number
  maxLat: number
}

export interface Polygon {
  type: "Polygon"
  coordinates: number[][]
}

export interface LineString {
  type: "LineString"
  coordinates: number[][]
}

export interface SpillObject {
  scene_id: string
  acquisition_timestamp: string
  confidence: number
  area_km2: number
  centroid: Point
  bbox: BBox
  polygon: Polygon
  major_axis_bearing: number
  elongation: number
}

export interface Alert {
  spill_id: string
  status: "new" | "update" | "possible" | "none"
  rules_fired: string[]
  first_seen: string
  last_updated: string
}

export interface Vessel {
  mmsi: string
  name: string
  vessel_type: string
  score: number
  explanation: string
  cpa_distance_km: number
  cpa_time: string
  track: LineString
}

export interface ForecastEntry {
  hours: 6 | 12 | 24 | 48
  time: string
  polygon: Polygon
}

export interface HindcastEntry {
  time: string
  polygon: Polygon
}

export interface DriftForecast {
  forecast: ForecastEntry[]
}

export interface DriftHindcast {
  hindcast: HindcastEntry[]
}

export interface ModelArchitecture {
  arch: string
  encoder: string
  in_channels: number
  num_classes: number
  class_names: string[]
  image_size: number
}

export interface ModelInfo {
  trained: boolean
  architecture?: ModelArchitecture
  train_samples?: number
  val_samples?: number
  mean_iou?: number
  oil_iou?: number
  look_alike_iou?: number
  precision?: number
  recall?: number
  dice?: number
  oil_as_lookalike_rate?: number
  lookalike_as_oil_rate?: number
  trained_at?: string
}

export interface Region {
  id: string
  label: string
  bbox: number[]
  scene_id: string | null
}

export interface Provenance {
  sar_source: string | null
  sar_scene_id: string | null
  ais_source_label: string | null
  wind_source: string | null
  current_source: string | null
}

export interface PipelineRun {
  spill: SpillObject
  alert: Alert
  vessels: Vessel[]
  drift: DriftForecast & DriftHindcast
  provenance: Provenance | null
}