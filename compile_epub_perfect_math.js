const fs = require('fs');
const path = require('path');
const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { SVG } = require('mathjax-full/js/output/svg.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');

const baseDir = '/home/nilesh/code/transformers-from-scratch';
const mathDir = path.join(baseDir, 'assets/math');
if (!fs.existsSync(mathDir)) {
  fs.mkdirSync(mathDir, { recursive: true });
}

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);

const tex = new TeX({ packages: AllPackages });
const svg = new SVG({ fontCache: 'none' });
const html = mathjax.document('', { InputJax: tex, OutputJax: svg });

console.log('Reading Transformers_From_Scratch.md...');
let md = fs.readFileSync(path.join(baseDir, 'Transformers_From_Scratch.md'), 'utf8');

// 1. Protect Code Blocks
const codeBlocks = [];
md = md.replace(/```[\s\S]*?```/g, (match) => {
  codeBlocks.push(match);
  return `%%CODEBLOCK_${codeBlocks.length - 1}%%`;
});

const inlineCodes = [];
md = md.replace(/`[^`\n]+`/g, (match) => {
  inlineCodes.push(match);
  return `%%INLINECODE_${inlineCodes.length - 1}%%`;
});

// Function to convert simple inline math to clean native HTML/Unicode
function tryConvertSimpleMathToHtml(mathStr) {
  let s = mathStr.trim();

  // Normalize spaces
  s = s.replace(/\\,/g, ' ').replace(/\\quad/g, '   ');

  // Exact mappings for common variables and symbols
  const exactMap = {
    'N': '<em>N</em>',
    'n': '<em>n</em>',
    'i': '<em>i</em>',
    'j': '<em>j</em>',
    'k': '<em>k</em>',
    'K': '<em>K</em>',
    'm': '<em>m</em>',
    'w': '<em>w</em>',
    'x': '<em>x</em>',
    'y': '<em>y</em>',
    'z': '<em>z</em>',
    't': '<em>t</em>',
    'p': '<em>p</em>',
    'q': '<em>q</em>',
    'P': '<em>P</em>',
    'Q': '<em>Q</em>',
    'V': '<em>V</em>',
    'e': '<em>e</em>',
    'c': '<em>c</em>',
    '\\ell': '<em>ℓ</em>',
    '\\ell-1': '<em>ℓ</em>−1',
    '\\ell - 1': '<em>ℓ</em>−1',
    '\\exp': 'exp',
    '\\ln': 'ln',
    '\\log': 'log',
    '\\max': 'max',
    '\\min': 'min',
    '\\in': '∈',
    '\\forall': '∀',
    '\\approx': '≈',
    '\\sim': '∼',
    '\\top': '<sup>⊤</sup>',
    '\\top \\top': '<sup>⊤</sup>',
    '\\odot': '⊙',
    '\\pm': '±',
    '\\le': '≤',
    '\\ge': '≥',
    '\\neq': '≠',
    '\\times': '×',
    '\\cdot': '·',
    '\\to': '→',
    '\\leftarrow': '←',
    '\\alpha': '<em>α</em>',
    '\\beta': '<em>β</em>',
    '\\gamma': '<em>γ</em>',
    '\\delta': '<em>δ</em>',
    '\\epsilon': '<em>ε</em>',
    '\\theta': '<em>θ</em>',
    '\\lambda': '<em>λ</em>',
    '\\mu': '<em>μ</em>',
    '\\sigma': '<em>σ</em>',
    '\\pi': '<em>π</em>',
    '\\Sigma': 'Σ',
    '\\mathbb{R}': 'ℝ',
    '\\mathbb{R}^d': 'ℝ<sup><em>d</em></sup>',
    '\\mathbb{R}^{d}': 'ℝ<sup><em>d</em></sup>',
    '\\mathbb{R}^{d_k}': 'ℝ<sup><em>d<sub>k</sub></em></sup>',
    '\\mathbb{R}^{d_v}': 'ℝ<sup><em>d<sub>v</sub></em></sup>',
    '\\mathbb{R}^{d_{\\text{model}}}': 'ℝ<sup><em>d</em><sub>model</sub></sup>',
    '\\mathbb{R}^{d_\\text{model}}': 'ℝ<sup><em>d</em><sub>model</sub></sup>',
    '\\mathbb{R}^{d_{\\text{ff}}}': 'ℝ<sup><em>d</em><sub>ff</sub></sup>',
    '\\mathbb{R}^{d_\\text{ff}}': 'ℝ<sup><em>d</em><sub>ff</sub></sup>',
    '\\mathcal{V}': '𝒱',
    '\\mathcal{C}': '𝒞',
    '\\mathcal{D}': '𝒟',
    '\\mathcal{S}': '𝒮',
    'O(1)': '<em>O</em>(1)',
    'O(N)': '<em>O</em>(<em>N</em>)',
    'O(N^2)': '<em>O</em>(<em>N</em><sup>2</sup>)',
    'd_k': '<em>d<sub>k</sub></em>',
    'd_v': '<em>d<sub>v</sub></em>',
    'd_{\\text{model}}': '<em>d</em><sub>model</sub>',
    'd_\\text{model}': '<em>d</em><sub>model</sub>',
    'd_{\\text{ff}}': '<em>d</em><sub>ff</sub>',
    'd_\\text{ff}': '<em>d</em><sub>ff</sub>',
    'W_E': '<strong>W</strong><sub><em>E</em></sub>',
    'W_Q': '<strong>W</strong><sub><em>Q</em></sub>',
    'W_K': '<strong>W</strong><sub><em>K</em></sub>',
    'W_V': '<strong>W</strong><sub><em>V</em></sub>',
    'W_O': '<strong>W</strong><sub><em>O</em></sub>',
    'W_U': '<strong>W</strong><sub><em>U</em></sub>',
    'W_{\\text{gate}}': '<strong>W</strong><sub>gate</sub>',
    'W_\\text{gate}': '<strong>W</strong><sub>gate</sub>',
    'W_{\\text{up}}': '<strong>W</strong><sub>up</sub>',
    'W_\\text{up}': '<strong>W</strong><sub>up</sub>',
    'W_{\\text{down}}': '<strong>W</strong><sub>down</sub>',
    'W_\\text{down}': '<strong>W</strong><sub>down</sub>',
    '\\mathbf{e}_i': '<strong>e</strong><sub><em>i</em></sub>',
    '\\mathbf{p}_i': '<strong>p</strong><sub><em>i</em></sub>',
    '\\mathbf{w}': '<strong>w</strong>',
    '\\mathbf{x}': '<strong>x</strong>',
    '\\mathbf{y}': '<strong>y</strong>',
    '\\mathbf{u}': '<strong>u</strong>',
    '\\mathbf{h}_i^{(0)}': '<strong>h</strong><sub><em>i</em></sub><sup>(0)</sup>',
    '\\mathbf{h}_i^{(\\ell)}': '<strong>h</strong><sub><em>i</em></sub><sup>(<em>ℓ</em>)</sup>',
    '\\mathbf{h}_i^{(\\ell-1)}': '<strong>h</strong><sub><em>i</em></sub><sup>(<em>ℓ</em>−1)</sup>',
    '\\mathbf{h}_j^{(\\ell-1)}': '<strong>h</strong><sub><em>j</em></sub><sup>(<em>ℓ</em>−1)</sup>',
    '\\alpha_{ij}^{(\\ell)}': '<em>α</em><sub><em>ij</em></sub><sup>(<em>ℓ</em>)</sup>',
    '\\alpha_{ij}': '<em>α</em><sub><em>ij</em></sub>',
    '\\pi_\\theta': '<em>π<sub>θ</sub></em>',
    '\\pi_{\\text{ref}}': '<em>π</em><sub>ref</sub>',
    '\\pi_\\text{ref}': '<em>π</em><sub>ref</sub>',
  };

  if (exactMap[s]) {
    return exactMap[s];
  }

  // Pure numbers
  if (/^[0-9]+(\.[0-9]+)?$/.test(s)) {
    return s;
  }

  // Single letters: $A$, $x$, etc.
  if (/^[A-Za-z]$/.test(s)) {
    return `<em>${s}</em>`;
  }

  // Letter with single subscript: $x_i$, $h_j$, $s_k$, $u_k$
  if (/^([A-Za-z])_([A-Za-z0-9])$/.test(s)) {
    const m = s.match(/^([A-Za-z])_([A-Za-z0-9])$/);
    return `<em>${m[1]}</em><sub>${m[2]}</sub>`;
  }

  // Vector bold letter with subscript: \mathbf{h}_i, \mathbf{e}_k
  if (/^\\mathbf\{([A-Za-z])\}_([A-Za-z0-9])$/.test(s)) {
    const m = s.match(/^\\mathbf\{([A-Za-z])\}_([A-Za-z0-9])$/);
    return `<strong>${m[1]}</strong><sub>${m[2]}</sub>`;
  }

  // Simple equality / range: k=1, N=10, i=1
  if (/^([A-Za-z])\s*=\s*([0-9]+)$/.test(s)) {
    const m = s.match(/^([A-Za-z])\s*=\s*([0-9]+)$/);
    return `<em>${m[1]}</em> = ${m[2]}`;
  }

  // Simple element of: c \in \mathcal{C}, w \in \mathcal{V}
  if (/^([A-Za-z])\s*\\in\s*\\mathcal\{([A-Za-z])\}$/.test(s)) {
    const m = s.match(/^([A-Za-z])\s*\\in\s*\\mathcal\{([A-Za-z])\}$/);
    return `<em>${m[1]}</em> ∈ &#x1D49E;` ; // mathcal C
  }

  return null; // Must be rendered via SVG
}

// 2. Convert Display Math $$ ... $$ to Clean SVG with bounded max-height
let displayCount = 0;
md = md.replace(/\$\$([\s\S]*?)\$\$/g, (match, mathStr) => {
  displayCount++;
  const filename = `display_${displayCount.toString().padStart(4, '0')}.svg`;
  const filePath = path.join(mathDir, filename);
  
  try {
    const node = html.convert(mathStr.trim(), { display: true });
    let svgContent = adaptor.innerHTML(node);
    
    // Remove ex dimensions and enforce responsive viewBox scaling
    svgContent = svgContent.replace(/width="[^"]+"/, 'width="100%"')
                           .replace(/height="[^"]+"/, 'height="auto"');
    
    fs.writeFileSync(filePath, svgContent, 'utf8');
    return `\n\n<div class="math-display" style="text-align: center; margin: 1.4em 0; page-break-inside: avoid;"><img src="assets/math/${filename}" class="math-display-img" style="display: block; margin: 0 auto; max-width: 90% !important; max-height: 3.5em !important; height: auto;" alt="Equation ${displayCount}" /></div>\n\n`;
  } catch (err) {
    console.error(`Failed display math ${displayCount}:`, err.message);
    return match;
  }
});
console.log(`Converted ${displayCount} display math blocks.`);

// 3. Convert Inline Math $ ... $
let inlineHtmlCount = 0;
let inlineSvgCount = 0;

md = md.replace(/\$([^\$\n]+?)\$/g, (match, mathStr) => {
  const htmlVersion = tryConvertSimpleMathToHtml(mathStr);
  if (htmlVersion) {
    inlineHtmlCount++;
    return htmlVersion;
  }

  // For complex inline math, generate clean SVG with tight inline style
  inlineSvgCount++;
  const filename = `inline_${inlineSvgCount.toString().padStart(4, '0')}.svg`;
  const filePath = path.join(mathDir, filename);

  try {
    const node = html.convert(mathStr.trim(), { display: false });
    let svgContent = adaptor.innerHTML(node);

    // Replace ex dimensions with viewBox-only responsive scale
    svgContent = svgContent.replace(/width="[^"]+"/, 'width="auto"')
                           .replace(/height="[^"]+"/, 'height="1.15em"');

    fs.writeFileSync(filePath, svgContent, 'utf8');
    // Enforce explicit inline styling directly on <img> so Kindle CANNOT blow it up!
    return `<img src="assets/math/${filename}" class="math-inline-img" style="height: 1.15em !important; max-height: 1.2em !important; width: auto !important; vertical-align: -0.2em !important; display: inline-block !important; margin: 0 2px !important;" alt="${mathStr.replace(/"/g, '&quot;')}" />`;
  } catch (err) {
    return match;
  }
});

console.log(`Converted ${inlineHtmlCount} inline math items directly to native HTML/Unicode (zero scaling bugs!).`);
console.log(`Converted ${inlineSvgCount} complex inline math items to bounded SVGs.`);

// 4. Restore Code Blocks
md = md.replace(/%%INLINECODE_(\d+)%%/g, (match, idx) => inlineCodes[parseInt(idx)]);
md = md.replace(/%%CODEBLOCK_(\d+)%%/g, (match, idx) => codeBlocks[parseInt(idx)]);

// 5. Replace '---' with '***' to completely prevent YAML frontmatter alias triggers in Pandoc
const lines = md.split('\n');
const cleanedLines = lines.map(l => (l.trim() === '---' ? '***' : l));
md = cleanedLines.join('\n');

const outMdPath = path.join(baseDir, 'manuscript_perfect_math.md');
fs.writeFileSync(outMdPath, md, 'utf8');
console.log(`Saved finalized manuscript to ${outMdPath} (${md.length} chars).`);
