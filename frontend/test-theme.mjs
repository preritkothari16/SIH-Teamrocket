// Quick smoke test: verify theme CSS variables are applied
import { chromium } from 'playwright';

const BASE = 'http://localhost:5173';

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  const errors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });

  await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.waitForTimeout(1500);

  // Check CSS variables are defined on :root
  const vars = await page.evaluate(() => {
    const root = document.documentElement;
    const style = getComputedStyle(root);
    return {
      spaceBlack: style.getPropertyValue('--space-black').trim(),
      oceanDeep: style.getPropertyValue('--ocean-deep').trim(),
      oceanBright: style.getPropertyValue('--ocean-bright').trim(),
      earthGreen: style.getPropertyValue('--earth-green').trim(),
      earthBright: style.getPropertyValue('--earth-bright').trim(),
      signalAmber: style.getPropertyValue('--signal-amber').trim(),
      signalRed: style.getPropertyValue('--signal-red').trim(),
      bgPrimary: style.getPropertyValue('--bg-primary').trim(),
      bgPanel: style.getPropertyValue('--bg-panel').trim(),
      textPrimary: style.getPropertyValue('--text-primary').trim(),
      accentPrimary: style.getPropertyValue('--accent-primary').trim(),
    };
  });

  console.log('=== Theme CSS Variables ===');
  for (const [k, v] of Object.entries(vars)) {
    const ok = v.length > 0;
    console.log(`  ${ok ? 'OK' : 'MISSING'} ${k}: ${v || '(empty)'}`);
  }

  // Check body background uses space-black
  const bodyBg = await page.evaluate(() => {
    const body = document.body;
    return getComputedStyle(body).backgroundColor;
  });
  console.log(`\n=== Body Background ===`);
  console.log(`  ${bodyBg}`);

  // Check panels exist
  const panels = await page.$$('.bg-bg-panel, [class*="bg-bg-panel"]');
  console.log(`\n=== Panel Elements ===`);
  console.log(`  Found ${panels.length} panel(s) with bg-bg-panel class`);

  // Check for console errors
  console.log(`\n=== Console Errors ===`);
  if (errors.length === 0) {
    console.log('  None');
  } else {
    errors.forEach((e) => console.log(`  ERROR: ${e}`));
  }

  // Screenshot
  await page.screenshot({ path: 'theme-test-screenshot.png', fullPage: true });
  console.log('\nScreenshot saved: theme-test-screenshot.png');

  await browser.close();
  console.log('\nDone.');
}

main().catch((e) => {
  console.error('Test failed:', e);
  process.exit(1);
});
