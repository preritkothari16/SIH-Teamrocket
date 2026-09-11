// Shared HTML-escaping for backend-sourced strings interpolated into
// innerHTML template strings. Same bug class src/output/map.py's popups were
// already fixed for (html.escape()) — kept in one place here so the frontend
// and backend fixes don't drift apart the way map.py and report.py once did.
//
// Only for genuinely untrusted values (AIS-derived vessel fields, free-text
// ids, etc). Markup a component generates itself (static strings, numeric/
// date formatting) is not run through this — escaping already-safe markup
// would corrupt it instead of protecting anything.
export function escapeHtml(value: unknown): string {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
