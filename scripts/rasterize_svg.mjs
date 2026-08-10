// Rasterize an SVG file to PNG via headless chromium — for figures whose SVG
// embeds raster panels (e.g. the scaffold-composed Figure 1). Standard SVG only
// (no live dashboard needed). Playwright resolves from the loom worktree, same
// as scripts/render_loom_svgs.mjs.
//
//   node scripts/rasterize_svg.mjs <in.svg> <out.png> [scale=2]
import { createRequire } from 'module';
import { readFileSync, writeFileSync } from 'fs';
const require = createRequire(
  '/Users/eranagmon/code/vivarium-workbench--loom-polish/vivarium_workbench/loom/package.json');
const { chromium } = require('playwright');

const [, , inPath, outPath, scaleArg] = process.argv;
if (!inPath || !outPath) { console.error('usage: rasterize_svg.mjs <in.svg> <out.png> [scale]'); process.exit(2); }
const scale = Number(scaleArg) || 2;
const svg = readFileSync(inPath, 'utf-8');
const vb = svg.match(/viewBox="0 0 ([0-9.]+) ([0-9.]+)"/);
const wh = svg.match(/width="([0-9.]+)"[^>]*height="([0-9.]+)"/);
const m = vb || wh;
const W = Math.round(Number(m[1])), H = Math.round(Number(m[2]));

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: scale });
await page.setContent(`<!doctype html><style>*{margin:0;padding:0}</style>${svg}`, { waitUntil: 'networkidle' });
await page.waitForTimeout(600);
const el = await page.$('svg');
const buf = await el.screenshot({ omitBackground: false });
writeFileSync(outPath, buf);
console.log(`rasterized ${outPath} (${W}x${H} @${scale}x, ${Math.round(buf.length / 1024)} KB)`);
await browser.close();
