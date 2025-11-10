import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Listen on all interfaces (needed for Docker)
    port: 3000,
    strictPort: true, // Fail if port is already in use
    watch: {
      usePolling: true, // Needed for Docker file watching on some systems
    },
  },
});
