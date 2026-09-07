import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { SpillObject, Vessel, DriftForecast, DriftHindcast } from '../types/schema';

type GeoFeature = GeoJSON.Feature;
type GeoFeatureCollection = GeoJSON.FeatureCollection;

const DEFAULT_CENTER: [number, number] = [20, 0];
const DEFAULT_ZOOM = 3;

const SPILL_STYLE: L.PathOptions = {
  color: '#ef4444',
  weight: 2,
  fillColor: '#ef4444',
  fillOpacity: 0.3,
};

const VESSEL_COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f97316', '#84cc16'];

const DRIFT_FORECAST_STYLE: L.PathOptions = {
  color: '#f59e0b',
  weight: 2,
  fillColor: '#f59e0b',
  fillOpacity: 0.15,
  dashArray: '5, 5',
};

const DRIFT_HINDCAST_STYLE: L.PathOptions = {
  color: '#6b7280',
  weight: 2,
  fillColor: '#6b7280',
  fillOpacity: 0.1,
  dashArray: '10, 5',
};

export class MapView {
  private map: L.Map | null = null;
  private container: HTMLElement;
  private spillLayer: L.GeoJSON | null = null;
  private vesselLayers: Map<string, L.LayerGroup> = new Map();
  private driftForecastLayer: L.GeoJSON | null = null;
  private driftHindcastLayer: L.GeoJSON | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.initMap();
  }

  private initMap(): void {
    this.map = L.map(this.container, {
      center: DEFAULT_CENTER,
      zoom: DEFAULT_ZOOM,
      zoomControl: true,
      attributionControl: true,
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19,
    }).addTo(this.map);
  }

  setSpill(spill: SpillObject): void {
    if (this.spillLayer) {
      this.map?.removeLayer(this.spillLayer);
    }

    const geojson = {
      type: 'Feature' as const,
      geometry: spill.polygon as unknown as GeoJSON.Geometry,
      properties: {
        scene_id: spill.scene_id,
        confidence: spill.confidence,
        area_km2: spill.area_km2,
      },
    };

    this.spillLayer = L.geoJSON(geojson, {
      style: SPILL_STYLE,
      onEachFeature: (_feature, layer) => {
        layer.bindPopup(this.createSpillPopup(spill));
      },
    }).addTo(this.map!);

    this.fitToSpill(spill);
  }

  private fitToSpill(spill: SpillObject): void {
    const bounds = L.geoJSON(spill.polygon as unknown as GeoJSON.Geometry).getBounds();
    this.map?.fitBounds(bounds, { padding: [50, 50], maxZoom: 12 });
  }

  private createSpillPopup(spill: SpillObject): string {
    return `
      <div class="p-2 min-w-[200px]">
        <h3 class="font-semibold text-gray-900">Oil Spill Detected</h3>
        <div class="mt-2 space-y-1 text-sm">
          <div><span class="font-medium">Scene:</span> ${spill.scene_id}</div>
          <div><span class="font-medium">Time:</span> ${new Date(spill.acquisition_timestamp).toLocaleString()}</div>
          <div><span class="font-medium">Confidence:</span> ${(spill.confidence * 100).toFixed(1)}%</div>
          <div><span class="font-medium">Area:</span> ${spill.area_km2.toFixed(2)} km²</div>
          <div><span class="font-medium">Elongation:</span> ${spill.elongation.toFixed(1)}</div>
          <div><span class="font-medium">Bearing:</span> ${spill.major_axis_bearing.toFixed(1)}°</div>
        </div>
      </div>
    `;
  }

  setVessels(vessels: Vessel[]): void {
    this.vesselLayers.forEach((layer) => this.map?.removeLayer(layer));
    this.vesselLayers.clear();

    vessels.forEach((vessel, index) => {
      const color = VESSEL_COLORS[index % VESSEL_COLORS.length];
      const layerGroup = L.layerGroup();

      const trackGeojson = {
        type: 'Feature' as const,
        geometry: vessel.track as unknown as GeoJSON.Geometry,
        properties: { mmsi: vessel.mmsi },
      };

      L.geoJSON(trackGeojson, {
        style: { color, weight: 3, opacity: 0.8 },
        pointToLayer: (_feature, latlng) => {
          return L.circleMarker(latlng, {
            radius: 6,
            fillColor: color,
            color: '#fff',
            weight: 2,
            fillOpacity: 1,
          });
        },
        onEachFeature: (_feature, layer) => {
          layer.bindPopup(this.createVesselPopup(vessel));
        },
      }).addTo(layerGroup);

      const startCoord = vessel.track.coordinates[0];
      if (startCoord) {
        const marker = L.marker([startCoord[1], startCoord[0]], {
          icon: L.divIcon({
            className: 'vessel-marker',
            html: `<div style="background:${color};width:12px;height:12px;border-radius:50%;border:2px solid white;box-shadow:0 2px 4px rgba(0,0,0,0.3)"></div>`,
            iconSize: [12, 12],
            iconAnchor: [6, 6],
          }),
        }).bindPopup(this.createVesselPopup(vessel));
        marker.addTo(layerGroup);
      }

      layerGroup.addTo(this.map!);
      this.vesselLayers.set(vessel.mmsi, layerGroup);
    });
  }

  private createVesselPopup(vessel: Vessel): string {
    return `
      <div class="p-2 min-w-[220px]">
        <h3 class="font-semibold text-gray-900">${vessel.name}</h3>
        <div class="mt-2 space-y-1 text-sm">
          <div><span class="font-medium">MMSI:</span> ${vessel.mmsi}</div>
          <div><span class="font-medium">Type:</span> ${vessel.vessel_type}</div>
          <div><span class="font-medium">Score:</span> ${(vessel.score * 100).toFixed(0)}%</div>
          <div><span class="font-medium">CPA Distance:</span> ${vessel.cpa_distance_km.toFixed(1)} km</div>
          <div><span class="font-medium">CPA Time:</span> ${new Date(vessel.cpa_time).toLocaleString()}</div>
          <div class="mt-1 text-xs text-gray-600">${vessel.explanation}</div>
        </div>
      </div>
    `;
  }

  setDrift(forecast: DriftForecast, hindcast: DriftHindcast): void {
    if (this.driftForecastLayer) {
      this.map?.removeLayer(this.driftForecastLayer);
    }
    if (this.driftHindcastLayer) {
      this.map?.removeLayer(this.driftHindcastLayer);
    }

    if (forecast.forecast.length > 0) {
      const features: GeoFeature[] = forecast.forecast.map((entry, i) => ({
        type: 'Feature',
        geometry: entry.polygon as unknown as GeoJSON.Geometry,
        properties: { hours: entry.hours, time: entry.time, index: i },
      }));

      const collection: GeoFeatureCollection = { type: 'FeatureCollection', features };
      this.driftForecastLayer = L.geoJSON(collection, {
        style: DRIFT_FORECAST_STYLE,
        onEachFeature: (feature, layer) => {
          layer.bindPopup(`
            <div class="p-2">
              <h4 class="font-medium">Drift Forecast</h4>
              <div class="text-sm mt-1">+${feature.properties?.hours}h: ${new Date(feature.properties?.time as string).toLocaleString()}</div>
            </div>
          `);
        },
      }).addTo(this.map!);
    }

    if (hindcast.hindcast.length > 0) {
      const features: GeoFeature[] = hindcast.hindcast.map((entry) => ({
        type: 'Feature',
        geometry: entry.polygon as unknown as GeoJSON.Geometry,
        properties: { time: entry.time },
      }));

      const collection: GeoFeatureCollection = { type: 'FeatureCollection', features };
      this.driftHindcastLayer = L.geoJSON(collection, {
        style: DRIFT_HINDCAST_STYLE,
        onEachFeature: (feature, layer) => {
          layer.bindPopup(`
            <div class="p-2">
              <h4 class="font-medium">Drift Hindcast</h4>
              <div class="text-sm mt-1">${new Date(feature.properties?.time as string).toLocaleString()}</div>
            </div>
          `);
        },
      }).addTo(this.map!);
    }
  }

  highlightVessel(mmsi: string): void {
    this.vesselLayers.forEach((layer, key) => {
      const isSelected = key === mmsi;
      layer.eachLayer((l) => {
        if (l instanceof L.Path) {
          l.setStyle({ weight: isSelected ? 5 : 3, opacity: isSelected ? 1 : 0.8 });
        }
      });
    });
  }

  clear(): void {
    if (this.spillLayer) this.map?.removeLayer(this.spillLayer);
    this.vesselLayers.forEach((layer) => this.map?.removeLayer(layer));
    this.vesselLayers.clear();
    if (this.driftForecastLayer) this.map?.removeLayer(this.driftForecastLayer);
    if (this.driftHindcastLayer) this.map?.removeLayer(this.driftHindcastLayer);
    this.spillLayer = null;
    this.driftForecastLayer = null;
    this.driftHindcastLayer = null;
  }

  destroy(): void {
    this.clear();
    this.map?.remove();
    this.map = null;
  }

  getMap(): L.Map | null {
    return this.map;
  }
}