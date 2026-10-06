const fs = require('fs');
const path = require('path');
const katex = require('katex');

const baseDir = '/home/nilesh/code/transformers-from-scratch';
const mdPath = path.join(baseDir, 'Transformers_From_Scratch.md');
const katexCssPath = path.join(baseDir, 'node_modules/katex/dist/katex.min.css');

let md = fs.readFileSync(mdPath, 'utf8');
const katexCss = fs.readFileSync(katexCssPath, 'utf8');

console.log('Original Markdown length:', md.length);

// 1. Protect code blocks and diagrams
const codeBlocks = [];
md = md.replace(/```[\s\S]*?```/g, (match) => {
  codeBlocks.push(match);
  return `%%CODEBLOCK_${codeBlocks.length - 1}%%`;
});

// Protect inline code
const inlineCodes = [];
md = md.replace(/`[^`\n]+`/g, (match) => {
  inlineCodes.push(match);
  return `%%INLINECODE_${inlineCodes.length - 1}%%`;
});

// 2. Global Multi-line Display Math: $$ ... $$
let displayCount = 0;
let displayFails = 0;
md = md.replace(/\$\$([\s\S]*?)\$\$/g, (match, math) => {
  displayCount++;
  try {
    const rendered = katex.renderToString(math.trim(), {
      displayMode: true,
      throwOnError: false
    });
    return `\n\n<div class="math-display-wrapper">${rendered}</div>\n\n`;
  } catch (err) {
    displayFails++;
    console.error('Display Math Fail:', math.slice(0, 40), err.message);
    return match;
  }
});

// 3. Global Inline Math: $ ... $
let inlineCount = 0;
let inlineFails = 0;
md = md.replace(/\$([^\$\n]+?)\$/g, (match, math) => {
  inlineCount++;
  try {
    return katex.renderToString(math.trim(), {
      displayMode: false,
      throwOnError: false
    });
  } catch (err) {
    inlineFails++;
    console.error('Inline Math Fail:', math, err.message);
    return match;
  }
});

console.log(`Rendered ${displayCount} display equations (fails: ${displayFails})`);
console.log(`Rendered ${inlineCount} inline equations (fails: ${inlineFails})`);

// 4. Restore code blocks
md = md.replace(/%%INLINECODE_(\d+)%%/g, (match, idx) => {
  const code = inlineCodes[parseInt(idx)].slice(1, -1);
  return `<code>${code.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code>`;
});

md = md.replace(/%%CODEBLOCK_(\d+)%%/g, (match, idx) => {
  const rawBlock = codeBlocks[parseInt(idx)];
  const firstLineEnd = rawBlock.indexOf('\n');
  const lang = rawBlock.slice(3, firstLineEnd).trim();
  const content = rawBlock.slice(firstLineEnd + 1, -3);

  if (lang === 'mermaid') {
    return `<div class="mermaid">${content}</div>`;
  } else if (lang === 'xml' || lang === 'svg') {
    return `<div class="svg-container">${content}</div>`;
  } else {
    return `<pre class="code-block"><code>${content.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre>`;
  }
});

// 5. Structure & Tagging (Line-by-line formatting)
const lines = md.split('\n');
const htmlLines = [];
let inList = false;

for (let i = 0; i < lines.length; i++) {
  const line = lines[i];

  if (line.startsWith('# ')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`<h1>${line.slice(2).trim()}</h1>`);
  } else if (line.startsWith('## ')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`<h2>${line.slice(3).trim()}</h2>`);
  } else if (line.startsWith('### ')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`<h3>${line.slice(4).trim()}</h3>`);
  } else if (line.startsWith('#### 💡 Why?')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`<div class="card why-card"><h4>💡 Why?</h4>`);
  } else if (line.startsWith('#### 📖 Definition')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`</div><div class="card def-card"><h4>📖 Definition & Architecture</h4>`);
  } else if (line.startsWith('#### 📐 The Mathematics')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`</div><div class="card math-card"><h4>📐 The Mathematics & Working</h4>`);
  } else if (line.startsWith('#### 🌍 Real-World')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`</div><div class="card real-card"><h4>🌍 Real-World & NLP Behaviors</h4>`);
  } else if (line.startsWith('---')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`</div><hr/>`);
  } else if (line.startsWith('* ') || line.startsWith('- ')) {
    if (!inList) { htmlLines.push('<ul>'); inList = true; }
    const item = line.slice(2).trim()
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>');
    htmlLines.push(`<li>${item}</li>`);
  } else if (/^\d+\.\s/.test(line)) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    const num = line.match(/^\d+\./)[0];
    const text = line.replace(/^\d+\.\s/, '').trim()
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>');
    htmlLines.push(`<p class="num-item"><strong>${num}</strong> ${text}</p>`);
  } else if (line.trim().startsWith('<div') || line.trim().startsWith('<pre') || line.trim() === '') {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(line);
  } else {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    const p = line.trim()
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>');
    htmlLines.push(`<p>${p}</p>`);
  }
}

if (inList) htmlLines.push('</ul>');
let bodyHtml = htmlLines.join('\n');

// Ensure any remaining open card divs are closed
const openDivs = (bodyHtml.match(/<div class="card /g) || []).length;
const closeDivs = (bodyHtml.match(/<\/div><hr\/>/g) || []).length;
console.log(`Cards opened: ${openDivs}, Cards closed with HR: ${closeDivs}`);

// Inject High-Res Nano Banana Architectural Figures
const diagramInjections = [
  { target: '<h3>0.1', img: 'diagram_0_polysemy.jpg', caption: 'Figure 0.1: Static Word2Vec Context Collapse vs. Dynamic Contextual Trajectories' },
  { target: '<h3>1.3', img: 'diagram_1_rope.jpg', caption: 'Figure 1.3: Rotary Position Embedding (RoPE) — 2D Complex Subspace Rotation & Relative Distance Angle (m - n)θ' },
  { target: '<h3>2.2', img: 'diagram_2_attention.jpg', caption: 'Figure 2.1: Scaled Dot-Product Attention Engine — Projections, Compatibility Matrix, Scaling, Softmax & Value Blending' },
  { target: '<h3>3.4', img: 'diagram_3_kv_cache.jpg', caption: 'Figure 3.1: Autoregressive Causal Attention Masking & Constant-Time O(1) Decoding via KV-Cache VRAM Buffers' },
  { target: '<h3>4.1', img: 'diagram_4_residual_norm.jpg', caption: 'Figure 4.1: The Residual Stream Highway Architecture with Pre-LN RMSNorm Normalization Branches' },
  { target: '<h3>5.2', img: 'diagram_5_swiglu.jpg', caption: 'Figure 5.1: SwiGLU Gated Activation Architecture — Content Branch × Smooth Multiplicative Gating Branch' },
  { target: '<h3>6.2', img: 'diagram_6_sampling.jpg', caption: 'Figure 6.1: Token Logit Projection, Temperature Entropy Contrast, and Top-p Nucleus Probability Filtering' },
  { target: '<h3>7.1', img: 'diagram_7_post_training.jpg', caption: 'Figure 7.1: The Post-Training Alignment Hierarchy — SFT, Human Preference (RLHF/DPO), and Verifiable Reasoning RL (GRPO)' }
];

for (const diag of diagramInjections) {
  const imgTag = `
<div class="figure-container">
  <img src="assets/${diag.img}" alt="${diag.caption}" class="figure-img" />
  <div class="figure-caption">${diag.caption}</div>
</div>`;
  bodyHtml = bodyHtml.replace(diag.target, `${imgTag}\n${diag.target}`);
}

const finalHtml = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Transformers: From Scratch</title>
<style>
${katexCss}

@page {
  size: A4;
  margin: 18mm 16mm;
}

body {
  font-family: 'Bookerly', 'Georgia', 'DejaVu Serif', serif;
  font-size: 11pt;
  line-height: 1.6;
  color: #1a1a1a;
  background-color: #ffffff;
  margin: 0;
  padding: 0;
}

h1 {
  font-size: 2.2em;
  text-align: center;
  color: #0f172a;
  border-bottom: 3px solid #1e293b;
  padding-bottom: 10px;
  margin-top: 25px;
  margin-bottom: 6px;
}

h2 {
  font-size: 1.55em;
  color: #0f172a;
  border-bottom: 2px solid #cbd5e1;
  padding-bottom: 6px;
  margin-top: 35px;
  margin-bottom: 14px;
  page-break-after: avoid;
}

h3 {
  font-size: 1.25em;
  color: #1e3a8a;
  margin-top: 28px;
  margin-bottom: 10px;
  border-left: 5px solid #2563eb;
  padding-left: 10px;
  page-break-after: avoid;
}

h4 {
  font-size: 1.05em;
  margin-top: 10px;
  margin-bottom: 6px;
  page-break-after: avoid;
}

.card {
  border-radius: 6px;
  padding: 12px 16px;
  margin: 12px 0;
  page-break-inside: avoid;
}

.why-card {
  background: #fffbeb;
  border: 1.5px solid #f59e0b;
  border-left: 5px solid #d97706;
}

.def-card {
  background: #f8fafc;
  border: 1.5px solid #cbd5e1;
  border-left: 5px solid #64748b;
}

.math-card {
  background: #f0fdf4;
  border: 1.5px solid #86efac;
  border-left: 5px solid #16a34a;
}

.real-card {
  background: #eff6ff;
  border: 1.5px solid #93c5fd;
  border-left: 5px solid #2563eb;
}

.math-display-wrapper {
  margin: 14px 0;
  text-align: center;
  overflow-x: auto;
}

.figure-container {
  text-align: center;
  margin: 20px auto;
  page-break-inside: avoid;
  max-width: 95%;
}

.figure-img {
  width: 100%;
  max-width: 720px;
  height: auto;
  border-radius: 6px;
  border: 1.5px solid #cbd5e1;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.figure-caption {
  font-size: 0.88em;
  font-style: italic;
  color: #475569;
  margin-top: 6px;
}

.code-block {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 5px;
  padding: 10px 12px;
  font-family: 'Courier New', monospace;
  font-size: 0.85em;
  overflow-x: auto;
  page-break-inside: avoid;
}

ul {
  padding-left: 20px;
  margin: 6px 0;
}

li {
  margin-bottom: 4px;
}

p {
  margin: 6px 0;
}

hr {
  border: 0;
  border-top: 1px dashed #cbd5e1;
  margin: 25px 0;
}
</style>
</head>
<body>

${bodyHtml}

</body>
</html>`;

const outHtmlPath = path.join(baseDir, 'index.html');
fs.writeFileSync(outHtmlPath, finalHtml, 'utf8');
console.log('index.html written successfully.');

// Audit check: Search index.html for any unrendered LaTeX
const rawMatches = finalHtml.match(/\$\$[\s\S]*?\$\$/g) || [];
const inlineMatches = (finalHtml.match(/\$[^\$\n<]+?\$/g) || []).filter(m => !m.includes('katex'));
console.log('AUDIT CHECK: Unrendered display math in index.html:', rawMatches.length);
console.log('AUDIT CHECK: Unrendered inline math in index.html:', inlineMatches.length);
if (rawMatches.length > 0) {
  console.log('Remaining display math samples:', rawMatches.slice(0, 3));
}
if (inlineMatches.length > 0) {
  console.log('Remaining inline math samples:', inlineMatches.slice(0, 5));
}
