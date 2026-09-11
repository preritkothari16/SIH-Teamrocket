import { beforeEach, describe, expect, it } from 'vitest'
import type { Vessel } from '../types/schema'
import { VesselDrawer } from './VesselDrawer'

function makeVessel(overrides: Partial<Vessel> = {}): Vessel {
  return {
    mmsi: '367123452',
    name: 'TANKER Gamma',
    vessel_type: 'tanker',
    score: 0.82,
    explanation: 'Near CPA at acquisition time.',
    cpa_distance_km: 2.4,
    cpa_time: '2024-04-10T14:22:00+00:00',
    track: { type: 'LineString', coordinates: [[0, 0], [0.1, 0.1]] },
    ...overrides,
  }
}

function makeDrawerElement(): HTMLElement {
  const drawer = document.createElement('div')
  drawer.innerHTML = `
    <div id="vessel-drawer-content"></div>
    <button id="vessel-drawer-close"></button>
  `
  document.body.appendChild(drawer)
  return drawer
}

describe('VesselDrawer XSS escaping', () => {
  let drawer: VesselDrawer
  let el: HTMLElement

  beforeEach(() => {
    el = makeDrawerElement()
    drawer = new VesselDrawer(el)
  })

  it('renders a vessel name containing <script> escaped, not as live markup', () => {
    drawer.render([makeVessel({ name: '<script>alert(1)</script>' })])

    expect(el.innerHTML).not.toContain('<script>alert(1)</script>')
    expect(el.querySelectorAll('script')).toHaveLength(0)
    // The escaped text still reaches the user, just inert.
    expect(el.textContent).toContain('<script>alert(1)</script>')
  })

  it('renders a double quote in the vessel name as plain text, not broken markup', () => {
    // A quote inside text content (not an attribute) can't break anything on
    // its own; the DOM round-trips &quot; back to a literal " when read via
    // innerHTML, same as a real browser. What matters is that it stays text,
    // not new markup — exactly one vessel card, no stray elements.
    drawer.render([makeVessel({ name: 'M/V "Sea Ghost"' })])

    expect(el.textContent).toContain('M/V "Sea Ghost"')
    expect(el.querySelectorAll('.rank-badge')).toHaveLength(1)
  })

  it('escapes an ampersand in the vessel name', () => {
    drawer.render([makeVessel({ name: 'Oil & Gas Co' })])

    expect(el.innerHTML).toContain('Oil &amp; Gas Co')
    expect(el.textContent).toContain('Oil & Gas Co')
  })

  it('escapes mmsi, vessel_type, and explanation the same way', () => {
    drawer.render([
      makeVessel({
        mmsi: '<img src=x onerror=alert(1)>',
        vessel_type: '<b>tanker</b>',
        explanation: 'Matched on <i>speed</i> & course.',
      }),
    ])

    expect(el.querySelectorAll('img')).toHaveLength(0)
    expect(el.querySelectorAll('b')).toHaveLength(0)
    expect(el.innerHTML).toContain('&lt;img src=x onerror=alert(1)&gt;')
    expect(el.innerHTML).toContain('&lt;b&gt;tanker&lt;/b&gt;')
    expect(el.innerHTML).toContain('Matched on &lt;i&gt;speed&lt;/i&gt; &amp; course.')
  })

  it('still renders a normal vessel name unchanged', () => {
    drawer.render([makeVessel({ name: 'TANKER Gamma' })])

    expect(el.textContent).toContain('TANKER Gamma')
    expect(el.innerHTML).toContain('TANKER Gamma')
  })
})
