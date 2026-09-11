import { describe, expect, it } from 'vitest'
import {
  angularDistanceDeg,
  buildClusters,
  clusterThresholdDeg,
  type ValidatedMarker,
} from './SpillGlobe'

function marker(overrides: Partial<ValidatedMarker> = {}): ValidatedMarker {
  return {
    spillId: 'm',
    runIndex: 0,
    lat: 0,
    lon: 0,
    confidence: 0.8,
    status: 'none',
    areaKm2: 1,
    ...overrides,
  }
}

describe('angularDistanceDeg', () => {
  it('is zero for the same point', () => {
    expect(angularDistanceDeg(24.1, -89.9, 24.1, -89.9)).toBeCloseTo(0, 6)
  })

  it('is ~90 degrees a quarter of the way around the globe', () => {
    expect(angularDistanceDeg(0, 0, 0, 90)).toBeCloseTo(90, 3)
  })

  it('is symmetric', () => {
    const a = angularDistanceDeg(10, 20, 30, 40)
    const b = angularDistanceDeg(30, 40, 10, 20)
    expect(a).toBeCloseTo(b, 9)
  })
})

describe('clusterThresholdDeg', () => {
  it('shrinks as zoom increases — this is what makes zooming in decluster', () => {
    const wide = clusterThresholdDeg(1, 1000, 800)
    const tight = clusterThresholdDeg(4, 1000, 800)
    expect(tight).toBeLessThan(wide)
  })

  it('shrinks as the canvas grows (more pixels per degree)', () => {
    const small = clusterThresholdDeg(1, 500, 400)
    const large = clusterThresholdDeg(1, 2000, 1600)
    expect(large).toBeLessThan(small)
  })
})

describe('buildClusters', () => {
  it('keeps the real lat/lon unchanged on every member', () => {
    const markers = [marker({ spillId: 'a', lat: 24.111, lon: -89.999 })]
    const clusters = buildClusters(markers, 5)
    expect(clusters[0].members[0].lat).toBe(24.111)
    expect(clusters[0].members[0].lon).toBe(-89.999)
  })

  it('groups two markers within threshold into one cluster', () => {
    // Coordinates deliberately kept away from any whole-degree grid
    // boundary — buildClusters buckets by grid cell, so two points this
    // close but straddling a cell edge (e.g. -90.0 vs -90.01) would land
    // in different cells; that's an accepted, documented characteristic
    // of grid clustering, not what this test is checking.
    const markers = [
      marker({ spillId: 'a', lat: 24.13, lon: -90.17 }),
      marker({ spillId: 'b', lat: 24.14, lon: -90.18 }), // ~1.3km apart
    ]
    const clusters = buildClusters(markers, 1) // 1 degree threshold
    expect(clusters).toHaveLength(1)
    expect(clusters[0].members).toHaveLength(2)
  })

  it('keeps two far-apart markers as separate singleton clusters', () => {
    const markers = [
      marker({ spillId: 'gulf', lat: 24.11, lon: -89.99 }),   // Gulf of Mexico
      marker({ spillId: 'north-sea', lat: 55.24, lon: 3.99 }), // North Sea
    ]
    const clusters = buildClusters(markers, 1)
    expect(clusters).toHaveLength(2)
    expect(clusters.every((c) => c.members.length === 1)).toBe(true)
  })

  it('re-splits a cluster into singletons once the threshold shrinks enough (zoom-in effect)', () => {
    const markers = [
      marker({ spillId: 'a', lat: 24.13, lon: -90.17 }),
      marker({ spillId: 'b', lat: 24.58, lon: -90.63 }),
    ]
    const grouped = buildClusters(markers, 2) // wide threshold — world view
    expect(grouped).toHaveLength(1)

    const split = buildClusters(markers, 0.1) // tight threshold — zoomed in
    expect(split).toHaveLength(2)
  })

  it('a 5-member cluster reports a member count and a dominant status', () => {
    const markers = [
      marker({ spillId: 'a', lat: 24.10, lon: -90.17, status: 'none' }),
      marker({ spillId: 'b', lat: 24.11, lon: -90.18, status: 'possible' }),
      marker({ spillId: 'c', lat: 24.12, lon: -90.19, status: 'new' }), // most urgent
      marker({ spillId: 'd', lat: 24.13, lon: -90.20, status: 'update' }),
      marker({ spillId: 'e', lat: 24.14, lon: -90.21, status: 'none' }),
    ]
    const [cluster] = buildClusters(markers, 1)
    expect(cluster.members).toHaveLength(5)
    expect(cluster.dominantStatus).toBe('new')
  })

  it('does not chain a spread-out necklace of points into one cluster at a tight threshold', () => {
    // Regression case for a real bug found via live testing: an earlier
    // greedy nearest-*running-mean* implementation let each new point
    // compare against a centroid that kept drifting to stay in range, so
    // five points spanning ~0.6 degrees (a segment of a shipping lane,
    // not a real single incident) never split apart even near max zoom.
    // Grid bucketing has no such drift to exploit.
    const markers = [
      marker({ spillId: 'a', lat: 24.00, lon: -90.30 }),
      marker({ spillId: 'b', lat: 24.30, lon: -89.90 }),
      marker({ spillId: 'c', lat: 23.80, lon: -90.10 }),
      marker({ spillId: 'd', lat: 24.40, lon: -90.20 }),
      marker({ spillId: 'e', lat: 24.15, lon: -89.75 }),
    ]
    const atMaxZoomThreshold = buildClusters(markers, 0.5)
    expect(atMaxZoomThreshold.length).toBeGreaterThan(1)
  })

  it('spreadDeg is 0 for a singleton and > 0 once a second member joins', () => {
    const singleton = buildClusters([marker({ lat: 10, lon: 10 })], 5)
    expect(singleton[0].spreadDeg).toBeCloseTo(0, 4)

    const pair = buildClusters(
      [marker({ spillId: 'a', lat: 10, lon: 10 }), marker({ spillId: 'b', lat: 10.2, lon: 10.2 })],
      5,
    )
    expect(pair[0].spreadDeg).toBeGreaterThan(0)
  })
})
