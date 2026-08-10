// Render each paper-figure composite to an SVG via bigraph-loom (headless).
//
// Drives the running dashboard's loom (Download → svg export, which embeds fonts
// + KaTeX and writes UTF-8), saving one SVG per study under
// studies/<slug>/visualizations/. Run with the loom worktree as cwd so
// `import 'playwright'` resolves:
//   cd <loom-worktree>/vivarium_workbench/loom && node <abs>/render_loom_svgs.mjs
import { createRequire } from 'module';
import { mkdirSync, writeFileSync } from 'fs';
import { dirname } from 'path';
// playwright lives in the loom worktree's node_modules; resolve from there.
const require = createRequire(
  '/Users/eranagmon/code/vivarium-workbench--loom-polish/vivarium_workbench/loom/package.json');
const { chromium } = require('playwright');

const BASE = 'http://127.0.0.1:8099';
const WS = '/Users/eranagmon/code/spatio-flux';

// [studySlug, compositeId, svgStem, detail?]  — detail pins the loom detail
// tier for the render ('' = Auto). Layout is always computed at the `full` tier,
// so detail only changes card content, not spacing — compactness comes from the
// composite's own size (see fig-08's schematic 4x4 grid in the export script).
const JOBS = [
  ['fig-01', 'spatio_flux.composites.fig01a-draft-processes',        'fig01a-draft-processes'],
  ['fig-01', 'spatio_flux.composites.fig01b-multiscale-composite',    'fig01b-multiscale-composite'],
  ['fig-02', 'spatio_flux.composites.fig02-process-bigraph',          'fig02-process-bigraph'],
  ['fig-03', 'spatio_flux.composites.fig03a-process-graph',           'fig03a-process-graph'],
  ['fig-03', 'spatio_flux.composites.fig03b-composite-process',       'fig03b-composite-process'],
  // Fig 7 is one study with three panels/runs (7.1/7.2/7.3).
  ['fig-07', 'spatio_flux.composites.fig07-1-community-dfba',     'fig07-1-community-dfba'],
  ['fig-07', 'spatio_flux.composites.fig07-2-comets',            'fig07-2-comets'],
  ['fig-07', 'spatio_flux.composites.fig07-3-brownian-particles', 'fig07-3-brownian-particles'],
  ['fig-08',   'spatio_flux.composites.fig08-reference-model',      'fig08-reference-model'],
];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1700, height: 1200 }, deviceScaleFactor: 2 });
let ok = 0;
for (const [slug, id, name, detail] of JOBS) {
  const out = `${WS}/studies/${slug}/visualizations/${name}.svg`;
  // nopersist=1 → the render never writes layouts back (so it can't clobber the
  // user's saved default view). The composite's workspace default view (mode +
  // positions + detail + collapse) is applied on load; the render captures it.
  const url = `${BASE}/bigraph-loom/?id=${encodeURIComponent(id)}&tabs=explore,document&nopersist=1`
    + (detail ? `&detail=${detail}` : '');
  const outPng = `${WS}/studies/${slug}/visualizations/${name}.png`;
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForSelector('.react-flow__node', { timeout: 40000 });
    // Extra settle: applying the workspace default view can switch layout mode
    // and re-run layout + fitView, on top of the initial layout + font load.
    await page.waitForTimeout(5000);
    // Grab the SVG string + PNG data-URL from loom's headless export hooks.
    const svg = await page.evaluate(async () => {
      const fn = window.__loomExportSvg;
      return fn ? await fn() : null;
    });
    if (!svg) throw new Error('__loomExportSvg returned null');
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(out, svg, 'utf-8');
    // PNG twin (same framing/fonts, 2×). Best-effort — a PNG failure must not
    // drop the SVG we already wrote.
    let pngKB = 0;
    try {
      const png = await page.evaluate(async () => {
        const fn = window.__loomExportPng;
        return fn ? await fn() : null;
      });
      if (png && png.startsWith('data:image/png')) {
        const buf = Buffer.from(png.slice(png.indexOf(',') + 1), 'base64');
        writeFileSync(outPng, buf);
        pngKB = Math.round(buf.length / 1024);
      }
    } catch { /* png best-effort */ }
    console.log('OK  ', slug, name, `(svg ${Math.round(svg.length / 1024)} KB, png ${pngKB} KB)`);
    ok++;
  } catch (e) {
    console.log('FAIL', slug, name, String(e).split('\n')[0].slice(0, 140));
  }
}
await browser.close();
console.log(`rendered ${ok}/${JOBS.length} loom figures (svg + png)`);
