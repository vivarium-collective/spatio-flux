// Render each spatio-flux TEST-SUITE composite to native bigraph-loom SVG + PNG.
//
// Same approach as the paper figures (scripts/render_loom_svgs.mjs): drive the
// running dashboard's loom headlessly, applying each composite's saved workspace
// default view, and emit one SVG + one PNG per study under
// studies/<study>/visualizations/. Jobs come from the investigation's member
// studies (their baseline composite) — see scripts that write /tmp/ts_jobs.json,
// or pass a jobs-file path as argv[2].
//
//   cd <loom-worktree>/vivarium_workbench/loom && node <abs>/render_test_suite_loom.mjs [jobs.json]
import { createRequire } from 'module';
import { mkdirSync, writeFileSync, readFileSync } from 'fs';
import { dirname } from 'path';
const require = createRequire(
  '/Users/eranagmon/code/vivarium-workbench--loom-polish/vivarium_workbench/loom/package.json');
const { chromium } = require('playwright');

const BASE = 'http://127.0.0.1:8099';
const WS = '/Users/eranagmon/code/spatio-flux';
const JOBS = JSON.parse(readFileSync(process.argv[2] || '/tmp/ts_jobs.json', 'utf-8'));

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1700, height: 1200 }, deviceScaleFactor: 2 });
let ok = 0;
for (const [study, id, name] of JOBS) {
  const outSvg = `${WS}/studies/${study}/visualizations/${name}.svg`;
  const outPng = `${WS}/studies/${study}/visualizations/${name}.png`;
  // collapse=1 folds repeated array processes (dFBA[i,j], particles[…]) into one
  // representative — cleaner diagrams AND far smaller SVGs for the big composites.
  const url = `${BASE}/bigraph-loom/?id=${encodeURIComponent(id)}&tabs=explore,document&collapse=1&nopersist=1`;
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForSelector('.react-flow__node', { timeout: 40000 });
    await page.waitForTimeout(5000);   // layout + default view + font load
    const svg = await page.evaluate(async () => (window.__loomExportSvg ? await window.__loomExportSvg() : null));
    if (!svg) throw new Error('__loomExportSvg returned null');
    mkdirSync(dirname(outSvg), { recursive: true });
    writeFileSync(outSvg, svg, 'utf-8');
    let pngKB = 0;
    try {
      const png = await page.evaluate(async () => (window.__loomExportPng ? await window.__loomExportPng() : null));
      if (png && png.startsWith('data:image/png')) {
        const buf = Buffer.from(png.slice(png.indexOf(',') + 1), 'base64');
        writeFileSync(outPng, buf); pngKB = Math.round(buf.length / 1024);
      }
    } catch { /* png best-effort */ }
    console.log('OK  ', study, `(svg ${Math.round(svg.length / 1024)} KB, png ${pngKB} KB)`);
    ok++;
  } catch (e) {
    console.log('FAIL', study, String(e).split('\n')[0].slice(0, 120));
  }
}
await browser.close();
console.log(`rendered ${ok}/${JOBS.length} test-suite loom figures (svg + png)`);
