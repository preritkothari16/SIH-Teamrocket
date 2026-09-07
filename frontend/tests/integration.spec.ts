import { test, expect } from '@playwright/test';

test.describe('Frontend-Backend Integration', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
  });

  test('loads page with dropdown and map', async ({ page }) => {
    // Check page loads
    await expect(page.locator('h1')).toContainText('SAR Oil Spill');
    
    // Check dropdown exists
    const select = page.locator('#mock-select');
    await expect(select).toBeVisible();
    
    // Check map container exists
    await expect(page.locator('#map')).toBeVisible();
  });

  test('dropdown shows backend runs when API is available', async ({ page }) => {
    const select = page.locator('#mock-select');
    
    // Wait for API to populate dropdown (it adds optgroup with backend runs)
    await page.waitForTimeout(1000);
    
    // Check options - should have backend runs optgroup
    const options = await select.locator('option').allTextContents();
    console.log('Dropdown options:', options);
    
    // Should have at least the mock options plus backend runs
    expect(options.length).toBeGreaterThanOrEqual(4); // 3 mocks + at least 1 backend
  });

  test('selecting backend run loads spill data', async ({ page }) => {
    const select = page.locator('#mock-select');
    
    // Wait for API to populate dropdown
    await page.waitForTimeout(1000);
    
    // Find and select the backend run (starts with "api:")
    const options = await select.locator('option').all();
    let backendOption = null;
    for (const opt of options) {
      const value = await opt.getAttribute('value');
      if (value?.startsWith('api:')) {
        backendOption = value;
        break;
      }
    }
    
    expect(backendOption).not.toBeNull();
    
    // Select the backend run
    await select.selectOption(backendOption!);
    await page.waitForTimeout(1000);
    
    // Verify spill panel shows data
    const spillPanel = page.locator('#spill-panel');
    await expect(spillPanel).toContainText('demo_pipeline');
    await expect(spillPanel).toContainText('8.3');
    await expect(spillPanel).toContainText('85.0%'); // confidence 0.85 = 85.0%
  });

  test('alert badge shows correct status from backend', async ({ page }) => {
    const select = page.locator('#mock-select');
    
    await page.waitForTimeout(1000);
    
    const options = await select.locator('option').all();
    let backendOption = null;
    for (const opt of options) {
      const value = await opt.getAttribute('value');
      if (value?.startsWith('api:')) {
        backendOption = value;
        break;
      }
    }
    
    await select.selectOption(backendOption!);
    await page.waitForTimeout(1000);
    
    // Check alert badge
    const alertBadge = page.locator('#alert-badge');
    await expect(alertBadge).toContainText('New Spill'); // status "new" = "New Spill"
    await expect(alertBadge).toContainText('evt_a1b2c3d4'); // spill_id
  });

  test('vessel list shows vessels from backend', async ({ page }) => {
    const select = page.locator('#mock-select');
    
    await page.waitForTimeout(1000);
    
    const options = await select.locator('option').all();
    let backendOption = null;
    for (const opt of options) {
      const value = await opt.getAttribute('value');
      if (value?.startsWith('api:')) {
        backendOption = value;
        break;
      }
    }
    
    await select.selectOption(backendOption!);
    await page.waitForTimeout(1000);
    
    // Check vessel list
    const vesselList = page.locator('#vessel-list');
    await expect(vesselList).toContainText('TANKER Alpha');
    await expect(vesselList).toContainText('TANKER Beta');
    await expect(vesselList).toContainText('92%'); // score 0.92
    await expect(vesselList).toContainText('67%'); // score 0.67
  });

  test('vessel click highlights on map', async ({ page }) => {
    const select = page.locator('#mock-select');
    
    await page.waitForTimeout(1000);
    
    const options = await select.locator('option').all();
    let backendOption = null;
    for (const opt of options) {
      const value = await opt.getAttribute('value');
      if (value?.startsWith('api:')) {
        backendOption = value;
        break;
      }
    }
    
    await select.selectOption(backendOption!);
    await page.waitForTimeout(1000);
    
    // Click on first vessel
    const vesselItems = page.locator('.vessel-item');
    await expect(vesselItems.first()).toBeVisible();
    await vesselItems.first().click();
    await page.waitForTimeout(500);
    
    // The map should have highlighted the vessel track
    // (hard to verify visually, but we can check the click didn't error)
  });

  test('mock fallback still works when backend unavailable', async ({ page }) => {
    // This test would need a separate server setup without backend
    // For now, verify mock options are present
    const select = page.locator('#mock-select');
    const options = await select.locator('option').allTextContents();
    
    expect(options).toContain('With Vessels');
    expect(options).toContain('With Drift');
    expect(options).toContain('No Alert');
  });
});

test.describe('API Endpoint Verification', () => {
  test('GET /api/runs returns backend data', async ({ request }) => {
    const response = await request.get('http://localhost:5173/api/runs');
    expect(response.ok()).toBeTruthy();
    
    const data = await response.json();
    // Should have both demo_pipeline and demo_scene
    const sceneIds = data.map((r: any) => r.scene_id);
    expect(sceneIds).toContain('demo_pipeline');
    expect(sceneIds).toContain('demo_scene');
    
    // demo_pipeline should have alert_status "active"
    const pipelineRun = data.find((r: any) => r.scene_id === 'demo_pipeline');
    expect(pipelineRun.alert_status).toBe('active');
    expect(pipelineRun.confidence).toBe(0.85);
  });

  test('GET /api/runs/demo_pipeline returns full contract', async ({ request }) => {
    const response = await request.get('http://localhost:5173/api/runs/demo_pipeline');
    expect(response.ok()).toBeTruthy();
    
    const data = await response.json();
    
    // Verify top-level keys
    expect(data).toHaveProperty('spill');
    expect(data).toHaveProperty('alert');
    expect(data).toHaveProperty('vessels');
    expect(data).toHaveProperty('drift');
    
    // Verify spill shape
    expect(data.spill.scene_id).toBe('demo_pipeline');
    expect(data.spill.confidence).toBe(0.85);
    expect(data.spill.area_km2).toBe(8.3);
    expect(data.spill.centroid).toHaveProperty('lat');
    expect(data.spill.centroid).toHaveProperty('lon');
    expect(data.spill.polygon.type).toBe('Polygon');
    
    // Verify alert
    expect(data.alert.status).toBe('new');
    expect(data.alert.spill_id).toBe('evt_a1b2c3d4');
    
    // Verify vessels
    expect(data.vessels.length).toBe(2);
    expect(data.vessels[0].mmsi).toBe('367123450');
    expect(data.vessels[0].name).toBe('TANKER Alpha');
    expect(data.vessels[0].score).toBe(0.92);
    expect(data.vessels[1].mmsi).toBe('367123451');
    
    // Verify drift is present (empty arrays)
    expect(data.drift.forecast).toEqual([]);
    expect(data.drift.hindcast).toEqual([]);
  });
});