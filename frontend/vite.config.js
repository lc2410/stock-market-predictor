import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/search': 'http://127.0.0.1:5001',
      '/predict': 'http://127.0.0.1:5001',
      '/predict_stream': {
        target: 'http://127.0.0.1:5001',
        // SSE requires these proxy settings for correct streaming behaviour
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            // Disable buffering so SSE events arrive immediately
            proxyReq.setHeader('X-Accel-Buffering', 'no');
          });
        },
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
