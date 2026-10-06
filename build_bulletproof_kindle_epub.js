const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const baseDir = '/home/nilesh/code/transformers-from-scratch';
const pngMathDir = path.join(baseDir, 'assets/math_png');
if (!fs.existsSync(pngMathDir)) {
  fs.mkdirSync(pngMathDir, { recursive: true });
}

const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { SVG } = require('mathjax-full/js/output/svg.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');
const { Resvg } = require('@resvg/resvg-js');

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

// 2. Pre-render All 90 Display Math Equations to Crisp 300 DPI PNGs
let displayCount = 0;
md = md.replace(/\$\$([\s\S]*?)\$\$/g, (match, mathStr) => {
  displayCount++;
  const filename = `display_${displayCount.toString().padStart(4, '0')}.png`;
  const filePath = path.join(pngMathDir, filename);

  const cleanEq = mathStr.trim();
  const node = html.convert(cleanEq, { display: true });
  const svgContent = adaptor.innerHTML(node);

  // Height determination: complex multiline equations get more height
  let renderHeight = 52;
  if (cleanEq.includes('\\\\') || cleanEq.includes('\\begin{aligned}') || cleanEq.includes('\\frac')) {
    renderHeight = 85;
  }
  if (cleanEq.includes('\\begin{matrix}') || cleanEq.includes('\\begin{bmatrix}')) {
    renderHeight = 110;
  }

  // Render to 2x PNG for crispness on 300 ppi Kindle Paperwhite
  const resvg = new Resvg(svgContent, {
    fitTo: { mode: 'height', value: renderHeight * 2 }
  });
  const pngBuffer = resvg.render().asPng();
  fs.writeFileSync(filePath, pngBuffer);

  // Use explicit HTML height attribute so Kindle cannot stretch it!
  return `\n\n<div class="math-display" style="text-align: center; margin: 1.4em 0; page-break-inside: avoid;"><img src="assets/math_png/${filename}" height="${renderHeight}" style="height: ${renderHeight}px !important; max-height: ${renderHeight}px !important; max-width: 90% !important; width: auto !important; display: block; margin: 0 auto;" alt="Equation ${displayCount}" /></div>\n\n`;
});
console.log(`Rendered ${displayCount} display equations to PNG images!`);

// 3. Convert 100% of ALL Inline Math to Clean Native Unicode Text
const subMap = {
  '0':'₀','1':'₁','2':'₂','3':'₃','4':'₄','5':'₅','6':'₆','7':'₇','8':'₈','9':'₉',
  '+':'₊','-':'₋','(':'₍',')':'₎','a':'ₐ','e':'ₑ','h':'ₕ','i':'ᵢ','j':'ⱼ','k':'ₖ',
  'l':'ₗ','m':'ₘ','n':'ₙ','o':'ₒ','p':'ₚ','r':'ᵣ','s':'ₛ','t':'ₜ','u':'ᵤ','v':'ᵥ','x':'ₓ',
  'ℓ':'ₗ'
};

const supMap = {
  '0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹',
  '+':'⁺','-':'⁻','(':'⁽',')':'⁾','n':'ⁿ','i':'ⁱ','T':'ᵀ','d':'ᵈ','l':'ˡ','m':'ᵐ',
  'k':'ᵏ','t':'ᵗ','o':'ᵒ','*':'﹡','ℓ':'ˡ'
};

function inlineTexToUnicode(mText) {
  let s = mText.trim();

  // Strip text/formatting wrappers
  s = s.replace(/\\text\{([^\}]+)\}/g, '$1');
  s = s.replace(/\\mathrm\{([^\}]+)\}/g, '$1');
  s = s.replace(/\\mathbf\{([^\}]+)\}/g, '$1');
  s = s.replace(/\\boldsymbol\{([^\}]+)\}/g, '$1');
  s = s.replace(/\\mathcal\{([^\}]+)\}/g, '$1');

  // Math words
  s = s.replace(/\\softmax/g, 'softmax')
       .replace(/\\exp/g, 'exp')
       .replace(/\\ln/g, 'ln')
       .replace(/\\log/g, 'log')
       .replace(/\\max/g, 'max')
       .replace(/\\min/g, 'min');

  // Set and spaces
  s = s.replace(/\\mathbb\{R\}\^?\{?d\}?/g, 'ℝᵈ')
       .replace(/\\mathbb\{R\}\^?\{?d_k\}?/g, 'ℝ^(d_k)')
       .replace(/\\mathbb\{R\}\^?\{?d_v\}?/g, 'ℝ^(d_v)')
       .replace(/\\mathbb\{R\}\^?\{?d_model\}?/g, 'ℝ^(d_model)')
       .replace(/\\mathbb\{R\}/g, 'ℝ');

  // Operators
  s = s.replace(/\\times/g, '×')
       .replace(/\\cdot/g, '·')
       .replace(/\\approx/g, '≈')
       .replace(/\\sim/g, '∼')
       .replace(/\\pm/g, '±')
       .replace(/\\le/g, '≤')
       .replace(/\\ge/g, '≥')
       .replace(/\\neq/g, '≠')
       .replace(/\\in/g, '∈')
       .replace(/\\notin/g, '∉')
       .replace(/\\forall/g, '∀')
       .replace(/\\rightarrow/g, '→')
       .replace(/\\to/g, '→')
       .replace(/\\leftarrow/g, '←')
       .replace(/\\top/g, 'ᵀ')
       .replace(/\\odot/g, '⊙')
       .replace(/\\sum/g, 'Σ')
       .replace(/\\prod/g, 'Π')
       .replace(/\\infty/g, '∞')
       .replace(/-\infty/g, '−∞')
       .replace(/\\sqrt\{([^\}]+)\}/g, '√($1)')
       .replace(/\\sqrt/g, '√')
       .replace(/\\dots/g, '…')
       .replace(/\\ldots/g, '…')
       .replace(/\\cdots/g, '…');

  // Greek letters
  s = s.replace(/\\alpha/g, 'α')
       .replace(/\\beta/g, 'β')
       .replace(/\\gamma/g, 'γ')
       .replace(/\\delta/g, 'δ')
       .replace(/\\epsilon/g, 'ε')
       .replace(/\\theta/g, 'θ')
       .replace(/\\Theta/g, 'Θ')
       .replace(/\\lambda/g, 'λ')
       .replace(/\\mu/g, 'μ')
       .replace(/\\sigma/g, 'σ')
       .replace(/\\pi/g, 'π')
       .replace(/\\ell/g, 'ℓ')
       .replace(/\\phi/g, 'ϕ')
       .replace(/\\Phi/g, 'Φ')
       .replace(/\\Delta/g, 'Δ')
       .replace(/\\Omega/g, 'Ω');

  // Fractions: \frac{a}{b} -> (a / b)
  s = s.replace(/\\frac\{([^\}]+)\}\{([^\}]+)\}/g, '($1 / $2)');

  // Braces, spaces
  s = s.replace(/\\\{/g, '{').replace(/\\\}/g, '}').replace(/\\,/g, ' ').replace(/\\quad/g, ' ');
  s = s.replace(/\\mid/g, '|').replace(/\\vert/g, '|').replace(/\\;/g, ' ');

  // Subscripts
  s = s.replace(/_\{([0-9a-zA-Z\(\)\+\-ℓ]+)\}/g, (match, val) => {
    return val.split('').map(c => subMap[c] || c).join('');
  });
  s = s.replace(/_([0-9a-zA-Zℓ])/g, (match, val) => subMap[val] || ('_' + val));

  // Superscripts
  s = s.replace(/\^\{([0-9a-zA-Z\(\)\+\-ℓ\*]+)\}/g, (match, val) => {
    return val.split('').map(c => supMap[c] || c).join('');
  });
  s = s.replace(/\^([0-9a-zA-Z\(\)\+\-ℓ\*])/g, (match, val) => supMap[val] || ('^' + val));

  // Clean remaining braces or backslashes
  s = s.replace(/\{/g, '').replace(/\}/g, '').replace(/\\/g, '');

  return `<em>${s.trim()}</em>`;
}

let inlineCount = 0;
md = md.replace(/\$([^\$\n]+?)\$/g, (match, mathStr) => {
  inlineCount++;
  return inlineTexToUnicode(mathStr);
});
console.log(`Converted ${inlineCount} inline math expressions to 100% native Unicode/HTML text!`);

// 4. Restore Code Blocks
md = md.replace(/%%INLINECODE_(\d+)%%/g, (match, idx) => inlineCodes[parseInt(idx)]);
md = md.replace(/%%CODEBLOCK_(\d+)%%/g, (match, idx) => codeBlocks[parseInt(idx)]);

// 5. Replace '---' with '***' to avoid YAML frontmatter alias triggers in Pandoc
md = md.split('\n').map(l => (l.trim() === '---' ? '***' : l)).join('\n');

const outMd = path.join(baseDir, 'manuscript_bulletproof.md');
fs.writeFileSync(outMd, md, 'utf8');
console.log(`Saved bulletproof manuscript to ${outMd} (${md.length} chars).`);

// 6. Compile to EPUB via Pandoc
console.log('Compiling bulletproof EPUB with Pandoc...');
const epubOut = path.join(baseDir, 'The_Vibe_Coders_Guide_to_LLMs.epub');
execSync(`pandoc ${outMd} -o ${epubOut} --epub-cover-image=assets/cover.jpg --css=epub_math.css --toc --metadata title="The Vibe Coder’s Guide to LLMs" --metadata subtitle="How Transformers Actually Work: From Basic Math to Reasoning AI" --metadata language="en"`, { stdio: 'inherit' });

fs.copyFileSync(epubOut, path.join(baseDir, 'Transformers_From_Scratch.epub'));
fs.copyFileSync(epubOut, '/home/nilesh/.openclaw/workspace/The_Vibe_Coders_Guide_to_LLMs.epub');
fs.copyFileSync(epubOut, '/home/nilesh/.openclaw/workspace/Transformers_From_Scratch.epub');
console.log('EPUB successfully compiled and synchronized to workspace!');
