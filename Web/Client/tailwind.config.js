/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        trakora: {
          bg: "#0B0F19",
          panel: "#121A2B",
          card: "#182238",
          border: "#233354",
          pitch: "#1a472a",
          pitchLight: "#225935",
          accent: "#00F59B", // High-visibility neon tactical green
          accentBlue: "#00B2FE",
          accentAmber: "#FFB020",
          accentRed: "#FF4D4D",
          textMuted: "#8E9EB5",
        },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', "monospace"],
      },
    },
  },
  plugins: [],
};
