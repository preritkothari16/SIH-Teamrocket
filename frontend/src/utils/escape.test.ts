import { describe, expect, it } from 'vitest'
import { escapeHtml } from './escape'

describe('escapeHtml', () => {
  it('escapes a script tag so it cannot execute as markup', () => {
    expect(escapeHtml('<script>alert(1)</script>')).toBe(
      '&lt;script&gt;alert(1)&lt;/script&gt;',
    )
  })

  it('escapes double quotes so they cannot break out of an attribute', () => {
    expect(escapeHtml('M/V "Sea Ghost"')).toBe('M/V &quot;Sea Ghost&quot;')
  })

  it('escapes ampersands without double-escaping other entities', () => {
    expect(escapeHtml('Oil & Gas Co')).toBe('Oil &amp; Gas Co')
  })

  it('leaves plain text untouched', () => {
    expect(escapeHtml('TANKER Gamma')).toBe('TANKER Gamma')
  })
})
