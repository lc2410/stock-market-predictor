/**
 * Frontend E2E Tests (Playwright)
 * --------------------------------
 * These tests validate the core UI interactions of the MarketLens React app.
 * They run against the Vite dev server and do NOT require the Flask backend,
 * using route mocking to simulate API responses where needed.
 */
import { test, expect } from '@playwright/test';

test.describe('Frontend UI — Page Load', () => {

  test('Loads the homepage and verifies the title', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle('MarketLens');
  });

  test('Renders the MarketLens logo', async ({ page }) => {
    await page.goto('/');
    const logo = page.locator('.app-logo');
    await expect(logo).toBeVisible();
    await expect(logo).toHaveAttribute('alt', 'MarketLens Logo');
  });

  test('Renders the search bar and forecast button', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#tickerInput')).toBeVisible();
    await expect(page.locator('#predictBtn')).toBeVisible();
    await expect(page.locator('#predictBtn')).toHaveText('Get Forecast');
  });

  test('Renders the theme toggle button', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#themeToggle')).toBeVisible();
  });

});

test.describe('Frontend UI — Search Interactions', () => {

  test('Typing into the search bar updates its value', async ({ page }) => {
    await page.goto('/');
    const input = page.locator('#tickerInput');
    await input.fill('AAPL');
    await expect(input).toHaveValue('AAPL');
  });

  test('Clear button appears when text is entered', async ({ page }) => {
    await page.goto('/');
    const input = page.locator('#tickerInput');
    const clearBtn = page.locator('#clearSearchBtn');

    // Clear button should not exist when the input is empty
    await expect(clearBtn).toBeHidden();

    // Type something — clear button should appear
    await input.fill('MSFT');
    await expect(clearBtn).toBeVisible();
  });

  test('Clear button resets the search input', async ({ page }) => {
    await page.goto('/');
    const input = page.locator('#tickerInput');
    const clearBtn = page.locator('#clearSearchBtn');

    await input.fill('MSFT');
    await expect(clearBtn).toBeVisible();

    await clearBtn.click();
    await expect(input).toHaveValue('');
    await expect(clearBtn).toBeHidden();
  });

  test('Autocomplete dropdown appears on search input', async ({ page }) => {
    // Mock the /search API so we don't need the Flask backend running
    await page.route('**/search/**', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          { symbol: 'AAPL', name: 'Apple Inc.' },
          { symbol: 'AAPLX', name: 'Apple Fund' },
        ]),
      });
    });

    await page.goto('/');
    const input = page.locator('#tickerInput');
    await input.fill('AAP');

    // Wait for the debounce (300ms) + network to resolve
    const dropdown = page.locator('#autocompleteResults');
    await expect(dropdown).toBeVisible({ timeout: 5000 });

    // Verify suggestions rendered
    const items = page.locator('.autocomplete-item');
    await expect(items).toHaveCount(2);
    await expect(items.first().locator('.ac-sym')).toHaveText('AAPL');
  });

  test('Clicking an autocomplete suggestion fills the input', async ({ page }) => {
    await page.route('**/search/**', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          { symbol: 'TSLA', name: 'Tesla, Inc.' },
        ]),
      });
    });

    await page.goto('/');
    const input = page.locator('#tickerInput');
    await input.fill('TSL');

    const dropdown = page.locator('#autocompleteResults');
    await expect(dropdown).toBeVisible({ timeout: 5000 });

    // Click the suggestion
    await page.locator('.autocomplete-item').first().click();
    await expect(input).toHaveValue('TSLA');

    // Dropdown should close after selection
    await expect(dropdown).toBeHidden();
  });

});

test.describe('Frontend UI — Theme Toggle', () => {

  test('Toggles between light and dark mode', async ({ page }) => {
    await page.goto('/');
    const html = page.locator('html');
    const themeBtn = page.locator('#themeToggle');

    // Default is light mode
    await expect(html).toHaveAttribute('data-theme', 'light');

    // Toggle to dark
    await themeBtn.click();
    await expect(html).toHaveAttribute('data-theme', 'dark');

    // Toggle back to light
    await themeBtn.click();
    await expect(html).toHaveAttribute('data-theme', 'light');
  });

  test('Logo swaps based on the active theme', async ({ page }) => {
    await page.goto('/');
    const logo = page.locator('.app-logo');
    const themeBtn = page.locator('#themeToggle');

    // Light mode logo
    await expect(logo).toHaveAttribute('src', '/media/logos/light.png');

    // Switch to dark mode
    await themeBtn.click();
    await expect(logo).toHaveAttribute('src', '/media/logos/dark.png');
  });

});