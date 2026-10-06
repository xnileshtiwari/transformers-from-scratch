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

// 2. Convert Display Math $$ ... $$ to SVG
let displayCount = 0;
md = md.replace(/\$\$([\s\S]*?)\$\$/g, (match, mathStr) => {
  displayCount++;
  const filename = `display_${displayCount.toString().padStart(4, '0')}.svg`;
  const filePath = path.join(mathDir, filename);
  
  try {
    const node = html.convert(mathStr.trim(), { display: true });
    let svgContent = adaptor.innerHTML(node);
    // Add fill: currentColor for dark mode compatibility on Kindle
    svgContent = svgContent.replace(/<svg /, '<svg class="math-svg-display" ');
    fs.writeFileSync(filePath, svgContent, 'utf8');
    return `\n\n<div class="math-display"><img src="assets/math/${filename}" class="math-display-img" alt="Equation ${displayCount}" /></div>\n\n`;
  } catch (err) {
    console.error(`Failed display math ${displayCount}:`, err.message);
    return match;
  }
});
console.log(`Converted ${displayCount} display math blocks to standalone SVGs.`);

// 3. Convert Inline Math $ ... $ to SVG
let inlineCount = 0;
md = md.replace(/\$([^\$\n]+?)\$/g, (match, mathStr) => {
  inlineCount++;
  const filename = `inline_${inlineCount.toString().padStart(4, '0')}.svg`;
  const filePath = path.join(mathDir, filename);
  
  try {
    const node = html.convert(mathStr.trim(), { display: false });
    let svgContent = adaptor.innerHTML(node);
    svgContent = svgContent.replace(/<svg /, '<svg class="math-svg-inline" ');
    fs.writeFileSync(filePath, svgContent, 'utf8');
    return `<img src="assets/math/${filename}" class="math-inline-img" alt="${mathStr.replace(/"/g, '&quot;')}" />`;
  } catch (err) {
    console.error(`Failed inline math ${inlineCount}:`, err.message);
    return match;
  }
});
console.log(`Converted ${inlineCount} inline math expressions to standalone SVGs.`);

// 4. Restore Code Blocks
md = md.replace(/%%INLINECODE_(\d+)%%/g, (match, idx) => inlineCodes[parseInt(idx)]);
md = md.replace(/%%CODEBLOCK_(\d+)%%/g, (match, idx) => codeBlocks[parseInt(idx)]);

// 5. Clean up emoji headings to avoid hollow box glyphs on E-Ink
md = md.replace(/####\s*💡\s*/g, '#### 💡 ')
       .replace(/####\s*📖\s*/g, '#### 📖 ')
       .replace(/####\s*📐\s*/g, '#### 📐 ')
       .replace(/####\s*🌍\s*/g, '#### 🌍 ')
       .replace(/####\s*🚉\s*/g, '#### Station ');

const outMdPath = path.join(baseDir, 'manuscript_svg_math.md');
fs.writeFileSync(outMdPath, md, 'utf8');
console.log(`Saved SVG-math manuscript to ${outMdPath} (${md.length} chars).`);
