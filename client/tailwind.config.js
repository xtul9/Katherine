/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Moody, atmospheric palette inspired by twilight
        'void': {
          50: '#f4f3f5',
          100: '#e9e7eb',
          200: '#d3cfd7',
          300: '#b8b0bf',
          400: '#958a9f',
          500: '#7a6d85',
          600: '#645769',
          700: '#524856',
          800: '#463d49',
          900: '#1a171c',
          950: '#0d0c0e',
        },
        'ember': {
          50: '#fef6ee',
          100: '#fcecd8',
          200: '#f8d4b0',
          300: '#f3b67d',
          400: '#ec8f48',
          500: '#e77324',
          600: '#d9591a',
          700: '#b44318',
          800: '#8f371b',
          900: '#742f19',
          950: '#3e150b',
        },
        'aether': {
          50: '#f0f7fe',
          100: '#ddedfc',
          200: '#c2e0fb',
          300: '#98ccf7',
          400: '#67b0f1',
          500: '#4491eb',
          600: '#2f73df',
          700: '#265ecd',
          800: '#254da6',
          900: '#234384',
          950: '#1a2b50',
        }
      },
      fontFamily: {
        'display': ['Playfair Display', 'Georgia', 'serif'],
        'body': ['Source Sans 3', 'system-ui', 'sans-serif'],
        'mono': ['JetBrains Mono', 'Menlo', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out forwards',
        'slide-up': 'slideUp 0.4s ease-out forwards',
        'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
        'typing': 'typing 1.5s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '0.8' },
        },
        typing: {
          '0%, 100%': { opacity: '0.3' },
          '50%': { opacity: '1' },
        }
      }
    },
  },
  plugins: [],
}
