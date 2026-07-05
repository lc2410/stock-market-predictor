import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  retries: 1,
  use: {
    baseURL: 'http://localhost:5173',
    headless: true,
  },
  // Automatically start the Vite dev server before tests and shut it down after.
  // 'reuseExistingServer' means if you already have `npm run dev` running, it uses that instead.
  webServer: {
    command: 'npx vite --host 127.0.0.1 --port 5173',
    url: 'http://localhost:5173',
    reuseExistingServer: true,
    timeout: 30_000,
    stdout: 'pipe',
  },
  reporter: [['html', { outputFolder: 'playwright-report', open: 'never' }]],
});
