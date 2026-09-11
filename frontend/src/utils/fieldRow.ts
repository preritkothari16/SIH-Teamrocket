import { escapeHtml } from './escape';

// Single source of truth for a "label / value" HUD row — the typography
// ProvenancePanel and SpillOverlay drifted apart on (10px uppercase
// tracking-wider labels vs plain 12px sentence-case) was two components
// hand-rolling the same conceptual row independently. Route both through
// this instead of re-aligning the class strings by hand, or they'll drift
// again the next time either panel changes.
//
// Also escapes label/value by default (ProvenancePanel's provenance
// strings — ais_source_label is explicitly free-text per CLAUDE.md — were
// never covered by the innerHTML-escaping pass that fixed VesselDrawer/
// RunList/SpillOverlay; routing through here closes that gap for any
// caller without each one having to remember to do it itself).
export interface FieldRowOptions {
  /** Raw, trusted SVG markup — not escaped, since it's markup this codebase
   *  generates itself, not user data. */
  icon?: string;
  /** Value uses font-mono. Default true — matches every existing row. */
  mono?: boolean;
  /** Truncate a long value with an ellipsis and carry the full value in a
   *  title tooltip, instead of letting it overflow the row. */
  truncateValue?: boolean;
  /** Bottom border divider between rows (SpillOverlay's convention). Purely
   *  the divider — independent of `spacing`, so a last-in-group row can
   *  drop the border without also shrinking its padding. */
  divider?: boolean;
  /** Row vertical padding: 'cozy' (py-1.5, SpillOverlay's convention) or
   *  'compact' (py-1, ProvenancePanel's convention). Default 'compact'. */
  spacing?: 'cozy' | 'compact';
}

export function fieldRow(label: string, value: string, opts: FieldRowOptions = {}): string {
  const { icon = '', mono = true, truncateValue = false, divider = false, spacing = 'compact' } = opts;

  const rowClass = [
    'flex items-center justify-between gap-3',
    spacing === 'cozy' ? 'py-1.5' : 'py-1',
    divider ? 'border-b border-zinc-800/30' : '',
  ].filter(Boolean).join(' ');

  const valueClass = [
    'text-xs text-zinc-300',
    mono ? 'font-mono' : '',
    truncateValue ? 'text-right truncate max-w-[65%]' : '',
  ].filter(Boolean).join(' ');

  const safeLabel = escapeHtml(label);
  const safeValue = escapeHtml(value);
  const titleAttr = truncateValue ? ` title="${safeValue}"` : '';

  return `
    <div class="${rowClass}">
      <span class="text-[10px] text-zinc-500 uppercase tracking-wider flex items-center gap-1.5">${icon}${safeLabel}</span>
      <span class="${valueClass}"${titleAttr}>${safeValue}</span>
    </div>
  `;
}
