/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        hydro: {
          dark: '#0a0f18',
          card: '#111827',
          border: '#1f2937',
          cyan: '#00E5FF',
          accent: '#00FF66',
          yellow: '#FFD700',
          blue: '#00B0FF',
        }
      }
    },
  },
  plugins: [],
}
