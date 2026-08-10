// Render each paper-figure composite to an SVG via bigraph-loom (headless).
//
// Drives the running dashboard's loom (Download → svg export, which embeds fonts
// + KaTeX and writes UTF-8), saving one SVG per study under
// studies/<slug>/visualizations/. Run with the loom worktree as cwd so
// `import 'playwright'` resolves:
//   cd <loom-worktree>/vivarium_workbench/loom && node <abs>/render_loom_svgs.mjs
import { createRequire } from 'module';
import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'fs';
import { dirname } from 'path';
// playwright + lz-string live in the loom worktree's node_modules; resolve there.
const require = createRequire(
  '/Users/eranagmon/code/vivarium-workbench--loom-polish/vivarium_workbench/loom/package.json');
const { chromium } = require('playwright');
const LZ = require('lz-string');  // encode committed default views into ?view=

// Override for a dashboard on a different port / a worktree: VW_BASE / VW_WS.
const BASE = process.env.VW_BASE || 'http://127.0.0.1:8099';
const WS = process.env.VW_WS || '/Users/eranagmon/code/spatio-flux';

// [studySlug, compositeId, svgStem, flags?]
//   flags: { detail: '<tier>', collapse: true, hyperedges: true,
//            ports: 'none'|'plain'|'types', config: 'on'|'off', contract: 'on'|'off' }
// nopersist=1 always loads the composite's saved default view for POSITIONS;
// these per-panel flags are applied ON TOP and OVERRIDE the saved view — so a
// panel saved in one mode still renders collapsed / as hyperedges when flagged
// (e.g. Fig 2a is the process-bigraph view saved in process mode, re-read as
// hyperedges; fig07e collapses its 16 per-bin dFBA processes). `detail` pins the
// whole loom detail tier ('' = Auto); ports/config/contract force individual
// Detail-menu features. Layout is always computed at the `full` tier.
const JOBS = [
  ['fig-01', 'spatio_flux.composites.fig01a-draft-processes',      'fig01a-draft-processes'],
  ['fig-01', 'spatio_flux.composites.fig01b-multiscale-composite', 'fig01b-multiscale-composite'],
  ['fig-01', 'spatio_flux.composites.fig01c-study-workflow',       'fig01c-study-workflow'],
  // Fig 2: one composite rendered two ways — 2a Milner hyperedges, 2b processes.
  ['fig-02', 'spatio_flux.composites.fig02-process-bigraph',       'fig02a-hyperedges', { hyperedges: true }],
  ['fig-02', 'spatio_flux.composites.fig02-process-bigraph',       'fig02b-processes'],
  ['fig-03', 'spatio_flux.composites.fig03a-process-graph',        'fig03a-process-graph'],
  ['fig-03', 'spatio_flux.composites.fig03b-composite-process',    'fig03b-composite-process'],
  // Fig 7 is one study with three panels (7.1/7.2/7.3).
  ['fig-07', 'spatio_flux.composites.fig07-1-community-dfba',      'fig07-1-community-dfba'],
  // 7e (COMETS): collapse the 16 per-bin dFBA[i,j] into a single dFBA[*] ×16.
  ['fig-07', 'spatio_flux.composites.fig07-2-comets',             'fig07-2-comets', { collapse: true }],
  ['fig-07', 'spatio_flux.composites.fig07-3-brownian-particles', 'fig07-3-brownian-particles'],
  ['fig-08', 'spatio_flux.composites.fig08-reference-model',       'fig08-reference-model'],
];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1700, height: 1200 }, deviceScaleFactor: 2 });
let ok = 0;
for (const [slug, id, name, flags = {}] of JOBS) {
  const out = `${WS}/studies/${slug}/visualizations/${name}.svg`;
  // nopersist=1 → the render never writes layouts back (so it can't clobber the
  // user's saved default view). The composite's workspace default view (mode +
  // positions + detail + collapse) is applied on load; the render captures it.
  // Then the per-panel flags below are applied on top (they override the view).
  const params = ['tabs=explore,document', 'nopersist=1'];
  // Embed the composite's COMMITTED default view (positions + saved detail mix)
  // if one exists — this makes the render REPRODUCIBLE anywhere, independent of
  // the machine-local .pbg/loom-views. Loaded via ?view= (highest priority); the
  // per-panel flags below still override it (they apply after the view).
  const viewFile = `${WS}/investigations/paper-figures/loom-views/${id}.json`;
  if (existsSync(viewFile)) {
    try {
      const view = JSON.parse(readFileSync(viewFile, 'utf-8'));
      params.push('view=' + LZ.compressToEncodedURIComponent(JSON.stringify(view)));
    } catch { /* corrupt view file → fall back to the server default */ }
  }
  if (flags.detail) params.push(`detail=${flags.detail}`);
  if (flags.collapse) params.push('collapse=1');
  if (flags.hyperedges) params.push('hyperedges=1');
  if (flags.ports) params.push(`ports=${flags.ports}`);
  if (flags.config) params.push(`config=${flags.config}`);
  if (flags.contract) params.push(`contract=${flags.contract}`);
  const url = `${BASE}/bigraph-loom/?id=${encodeURIComponent(id)}&${params.join('&')}`;
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
