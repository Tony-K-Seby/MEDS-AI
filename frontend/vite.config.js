import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: './', // Important for relative path routing in production
  build: {
    outDir: 'dist',
  },
})
