import { expect, test } from '@playwright/test';

test.describe('Frontend-backend integration', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#loading-state')).toBeHidden();
  });

  test('renders backend runs in the globe dashboard', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'SAR Oil Spill' })).toBeVisible();
    await expect(page.locator('#globe-canvas')).toBeVisible();
    await expect(page.locator('#run-list .run-item')).not.toHaveCount(0);
    await expect(page.locator('#run-list')).toContainText('evt_a1b2c3d4');
  });

  test('selecting the demo pipeline run displays its spill and vessels', async ({ page }) => {
    await page.locator('.run-item[data-run-id="evt_a1b2c3d4"]').click();

    await expect(page.locator('#spill-overlay')).toContainText('demo_pipeline');
    await expect(page.locator('#spill-overlay')).toContainText('New Spill');
    await expect(page.locator('#spill-overlay')).toContainText('85.0%');
    await expect(page.locator('#spill-overlay')).toContainText('8.30');
    await expect(page.locator('#vessel-drawer-content')).toContainText('TANKER Alpha');
    await expect(page.locator('#vessel-drawer-content')).toContainText('TANKER Beta');
    await expect(page.locator('#report-btn')).toBeVisible();
  });
});

test.describe('API endpoint verification', () => {
  test('lists the committed demo run', async ({ request }) => {
    const response = await request.get('http://localhost:5173/api/runs');
    expect(response.ok()).toBeTruthy();

    const runs = await response.json();
    expect(runs).toEqual(expect.arrayContaining([
      expect.objectContaining({
        scene_id: 'demo_pipeline',
        alert_status: 'active',
        confidence: 0.85,
      }),
    ]));
  });

  test('returns the full demo pipeline contract', async ({ request }) => {
    const response = await request.get('http://localhost:5173/api/runs/demo_pipeline');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data.spill).toMatchObject({
      scene_id: 'demo_pipeline',
      confidence: 0.85,
      area_km2: 8.3,
    });
    expect(data.alert).toMatchObject({ status: 'new', spill_id: 'evt_a1b2c3d4' });
    expect(data.vessels).toEqual(expect.arrayContaining([
      expect.objectContaining({ name: 'TANKER Alpha', score: 0.92 }),
      expect.objectContaining({ name: 'TANKER Beta', score: 0.67 }),
    ]));
  });
});
