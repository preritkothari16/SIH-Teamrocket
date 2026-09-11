/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        /* Core palette from CSS vars */
        space: {
          DEFAULT: "var(--space-black)",
        },
        ocean: {
          deep: "var(--ocean-deep)",
          bright: "var(--ocean-bright)",
        },
        earth: {
          green: "var(--earth-green)",
          bright: "var(--earth-bright)",
        },
        signal: {
          red: "var(--signal-red)",
          amber: "var(--signal-amber)",
        },

        /* Semantic aliases */
        bg: {
          primary: "var(--bg-primary)",
          panel: "var(--bg-panel)",
        },
        border: {
          panel: "var(--border-panel)",
        },
        txt: {
          primary: "var(--text-primary)",
          muted: "var(--text-muted)",
        },
        accent: {
          primary: "var(--accent-primary)",
          secondary: "var(--accent-secondary)",
        },
        status: {
          new: "var(--status-new)",
          possible: "var(--status-possible)",
          clear: "var(--status-clear)",
        },

        /* Shadcn/ui compat aliases */
        background: "var(--bg-primary)",
        foreground: "var(--text-primary)",
        muted: {
          DEFAULT: "var(--ocean-deep)",
          foreground: "var(--text-muted)",
        },
        accent2: {
          DEFAULT: "var(--accent-primary)",
          foreground: "var(--text-primary)",
        },
        destructive: {
          DEFAULT: "var(--signal-red)",
          foreground: "var(--text-primary)",
        },
        ring: "var(--accent-primary)",
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
        display: ["Geist", "sans-serif"],
      },
      boxShadow: {
        "ocean-glow": "0 0 15px 3px color-mix(in srgb, var(--ocean-bright) 30%, transparent)",
        "red-glow": "0 0 12px 2px color-mix(in srgb, var(--signal-red) 40%, transparent)",
        "amber-glow": "0 0 12px 2px color-mix(in srgb, var(--signal-amber) 40%, transparent)",
      },
    },
  },
  plugins: [],
}
