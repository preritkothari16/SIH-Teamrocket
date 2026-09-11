import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'jsdom',
    // Scoped to src/ so this doesn't try to collect tests/integration.spec.ts
    // (the existing Playwright suite) as a vitest test.
    include: ['src/**/*.test.ts'],
  },
})
