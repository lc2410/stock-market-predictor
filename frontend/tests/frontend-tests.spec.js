import { test, expect } from '@playwright/test';

test.describe('Frontend UI Interactions', () => {
  
  test('Loads the homepage and verifies the title', async ({ page }) => {
    // The CI pipeline will run Flask on port 5001
    await page.goto('http://127.0.0.1:5001/');
    await expect(page).toHaveTitle('Stock & Dividend Forecaster');
  });

  test('Performs a search and displays the loader', async ({ page }) => {
    await page.goto('http://127.0.0.1:5001/');

    // Target the input and button based on your HTML IDs
    const tickerInput = page.locator('#tickerInput');
    const predictBtn = page.locator('#predictBtn');
    const loader = page.locator('#loader');

    // Type a ticker
    await tickerInput.fill('AAPL');
    await expect(tickerInput).toHaveValue('AAPL');

    // Click the forecast button
    await predictBtn.click();

    // Verify the UI locks down and shows the loading state
    await expect(loader).toBeVisible();
    await expect(tickerInput).toBeDisabled();
    await expect(predictBtn).toBeDisabled();
  });

  test('Clear search button works correctly', async ({ page }) => {
    await page.goto('http://127.0.0.1:5001/');

    const tickerInput = page.locator('#tickerInput');
    const clearBtn = page.locator('#clearSearchBtn');

    await tickerInput.fill('MSFT');
    await expect(clearBtn).toBeVisible();

    await clearBtn.click();
    await expect(tickerInput).toHaveValue('');
    await expect(clearBtn).toBeHidden();
  });
});