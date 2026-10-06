const fs = require('fs');
const path = require('path');
const katex = require('katex');

const baseDir = '/home/nilesh/code/transformers-from-scratch';
const mdPath = path.join(baseDir, 'Transformers_From_Scratch.md');
const katexCssPath = path.join(baseDir, 'node_modules/katex/dist/katex.min.css');

let md = fs.readFileSync(mdPath, 'utf8');
const katexCss = fs.readFileSync(katexCssPath, 'utf8');

// SVG Vector Icons for Badges
const ICONS = {
  why: `<svg class="sec-icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#d97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 0 0-7 7c0 2.5 1.5 4.5 3 6h8c1.5-1.5 3-3.5 3-6a7 7 0 0 0-7-7z"/></svg>`,
  def: `<svg class="sec-icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>`,
  math: `<svg class="sec-icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#16a34a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 12h10"/><path d="M12 7v10"/></svg>`,
  real: `<svg class="sec-icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>`,
  station: `<svg class="sec-icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#8b5cf6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 8v8"/><path d="M8 12h8"/></svg>`
};

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

// 2. Pre-render Display Math: $$ ... $$
const displayMathBlocks = [];
md = md.replace(/\$\$([\s\S]*?)\$\$/g, (match, mathStr) => {
  try {
    const rendered = katex.renderToString(mathStr.trim(), {
      displayMode: true,
      throwOnError: false
    });
    displayMathBlocks.push(`<div class="math-display-wrapper">${rendered}</div>`);
  } catch (err) {
    displayMathBlocks.push(`<div class="math-display-wrapper">${match}</div>`);
  }
  return `%%DISPLAYMATH_${displayMathBlocks.length - 1}%%`;
});

// 3. Pre-render Inline Math: $ ... $
const inlineMathBlocks = [];
md = md.replace(/\$([^\$\n]+?)\$/g, (match, mathStr) => {
  try {
    const rendered = katex.renderToString(mathStr.trim(), {
      displayMode: false,
      throwOnError: false
    });
    inlineMathBlocks.push(rendered);
  } catch (err) {
    inlineMathBlocks.push(match);
  }
  return `%%INLINEMATH_${inlineMathBlocks.length - 1}%%`;
});

// 4. Parse Markdown Images: ![caption](url) -> HTML <figure>
md = md.replace(/!\[(.*?)\]\((.*?)\)/g, (match, caption, src) => {
  return `\n\n<div class="figure-container">
  <img src="${src.trim()}" alt="${caption.trim()}" class="figure-img" />
  <div class="figure-caption">${caption.trim()}</div>
</div>\n\n`;
});

// 5. Line-by-line Structure
const lines = md.split('\n');
const htmlLines = [];
let inList = false;

for (let i = 0; i < lines.length; i++) {
  let line = lines[i];

  if (line.startsWith('# ')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`<h1>${line.slice(2).trim()}</h1>`);
  } else if (line.startsWith('## ')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`<h2>${line.slice(3).trim()}</h2>`);
  } else if (line.startsWith('### ')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`<h3>${line.slice(4).trim()}</h3>`);
  } else if (line.includes('#### 💡 Why?') || line.includes('#### Why?')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`<div class="card why-card"><div class="card-header">${ICONS.why} <span class="badge-title">MOTIVATION & FUNCTIONAL ROLE</span></div>`);
  } else if (line.includes('#### 📖 Definition') || line.includes('#### Definition')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`</div><div class="card def-card"><div class="card-header">${ICONS.def} <span class="badge-title">DEFINITION & ARCHITECTURE</span></div>`);
  } else if (line.includes('#### 📐 The Mathematics') || line.includes('#### The Mathematics')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`</div><div class="card math-card"><div class="card-header">${ICONS.math} <span class="badge-title">MATHEMATICAL WORKING & FORMULATION</span></div>`);
  } else if (line.includes('#### 🌍 Real-World') || line.includes('#### Real-World')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(`</div><div class="card real-card"><div class="card-header">${ICONS.real} <span class="badge-title">REAL-WORLD & NLP BEHAVIORS</span></div>`);
  } else if (line.startsWith('#### 🚉') || line.startsWith('#### Station')) {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    const title = line.replace(/####\s*(?:🚉|Station)?\s*/, '').trim();
    htmlLines.push(`<div class="station-header">${ICONS.station} <h4>${title}</h4></div>`);
  } else if (line.startsWith('---') || line.startsWith('***')) {
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
  } else if (line.trim().startsWith('<div') || line.trim().startsWith('<pre') || line.trim().startsWith('%%DISPLAYMATH_') || line.trim() === '') {
    if (inList) { htmlLines.push('</ul>'); inList = false; }
    htmlLines.push(line);
  } else if (line.trim().startsWith('|')) {
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

// Clean table rendering
bodyHtml = bodyHtml.replace(/(\|.+?\|\n\|[-:\s|]+?\|\n(?:\|.+?\|\n?)+)/g, (tableMarkdown) => {
  const rows = tableMarkdown.trim().split('\n');
  const headers = rows[0].split('|').slice(1, -1).map(c => c.trim());
  const bodyRows = rows.slice(2);

  let tableHtml = '<table class="styled-table">\n<thead>\n<tr>\n';
  headers.forEach(h => {
    tableHtml += `  <th>${h.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</th>\n`;
  });
  tableHtml += '</tr>\n</thead>\n<tbody>\n';

  bodyRows.forEach(r => {
    const cells = r.split('|').slice(1, -1).map(c => c.trim());
    tableHtml += '<tr>\n';
    cells.forEach(c => {
      tableHtml += `  <td>${c.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\*(.*?)\*/g, '<em>$1</em>')}</td>\n`;
    });
    tableHtml += '</tr>\n';
  });
  tableHtml += '</tbody>\n</table>\n';
  return tableHtml;
});

// Restore tokens safely using functional replacers
bodyHtml = bodyHtml.replace(/%%DISPLAYMATH_(\d+)%%/g, (match, idx) => displayMathBlocks[parseInt(idx)]);
bodyHtml = bodyHtml.replace(/%%INLINEMATH_(\d+)%%/g, (match, idx) => inlineMathBlocks[parseInt(idx)]);
bodyHtml = bodyHtml.replace(/%%INLINECODE_(\d+)%%/g, (match, idx) => {
  const code = inlineCodes[parseInt(idx)].slice(1, -1);
  return `<code>${code.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code>`;
});
bodyHtml = bodyHtml.replace(/%%CODEBLOCK_(\d+)%%/g, (match, idx) => {
  const rawBlock = codeBlocks[parseInt(idx)];
  const firstLineEnd = rawBlock.indexOf('\n');
  const lang = rawBlock.slice(3, firstLineEnd).trim();
  const content = rawBlock.slice(firstLineEnd + 1, -3);

  if (lang === 'mermaid') {
    return '';
  } else {
    return `<pre class="code-block"><code>${content.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre>`;
  }
});

function createHtmlDocument(isKindleLayout) {
  const pageCss = isKindleLayout ? `
    @page {
      size: 120mm 160mm;
      margin: 8mm 6mm;
    }
    body {
      font-size: 10pt;
      line-height: 1.5;
    }
    h1 { font-size: 1.8em; }
    h2 { font-size: 1.35em; }
    h3 { font-size: 1.15em; }
    .figure-img { max-width: 100%; }
  ` : `
    @page {
      size: A4;
      margin: 18mm 16mm;
    }
    body {
      font-size: 11pt;
      line-height: 1.6;
    }
    h1 { font-size: 2.2em; }
    h2 { font-size: 1.55em; }
    h3 { font-size: 1.25em; }
    .figure-img { max-width: 720px; }
  `;

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>The Vibe Coder’s Guide to LLMs</title>
<style>
${katexCss}

${pageCss}

body {
  font-family: 'Bookerly', 'Georgia', 'DejaVu Serif', serif;
  color: #1a1a1a;
  background-color: #ffffff;
  margin: 0;
  padding: 0;
}

.cover-page-wrapper {
  page-break-after: always;
  width: 100%;
  height: 100vh;
  margin: 0;
  padding: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  background: #030712;
}
.cover-img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

h1 {
  text-align: center;
  color: #0f172a;
  border-bottom: 3px solid #1e293b;
  padding-bottom: 10px;
  margin-top: 25px;
  margin-bottom: 6px;
}

h2 {
  color: #0f172a;
  border-bottom: 2px solid #cbd5e1;
  padding-bottom: 6px;
  margin-top: 35px;
  margin-bottom: 14px;
  page-break-after: avoid;
}

h3 {
  color: #1e3a8a;
  margin-top: 28px;
  margin-bottom: 10px;
  border-left: 5px solid #2563eb;
  padding-left: 10px;
  page-break-after: avoid;
}

.station-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 20px;
  margin-bottom: 8px;
  color: #7c3aed;
  page-break-after: avoid;
}
.station-header h4 {
  font-size: 1.1em;
  margin: 0;
  color: #6d28d9;
}

.card {
  border-radius: 6px;
  padding: 14px 18px;
  margin: 16px 0;
  page-break-inside: avoid;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  font-weight: 700;
  font-size: 0.9em;
  letter-spacing: 0.05em;
}

.sec-icon {
  display: inline-block;
  vertical-align: middle;
}

.why-card { background: #fffdf5; border: 1.5px solid #fde68a; border-left: 5px solid #d97706; }
.why-card .card-header { color: #b45309; }

.def-card { background: #f8fafc; border: 1.5px solid #cbd5e1; border-left: 5px solid #475569; }
.def-card .card-header { color: #334155; }

.math-card { background: #f0fdf4; border: 1.5px solid #bbf7d0; border-left: 5px solid #16a34a; }
.math-card .card-header { color: #15803d; }

.real-card { background: #eff6ff; border: 1.5px solid #bfdbfe; border-left: 5px solid #2563eb; }
.real-card .card-header { color: #1d4ed8; }

.math-display-wrapper {
  margin: 14px 0;
  text-align: center;
  overflow: visible !important;
  max-width: 100%;
}

.katex-display {
  overflow: visible !important;
  margin: 0.6em 0 !important;
}

.katex {
  font-size: 1.05em;
}

.figure-container {
  text-align: center;
  margin: 22px auto;
  page-break-inside: avoid;
  max-width: 96%;
}

.figure-img {
  width: 100%;
  height: auto;
  border-radius: 6px;
  border: 1.5px solid #cbd5e1;
  box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

.figure-caption {
  font-size: 0.88em;
  font-style: italic;
  color: #475569;
  margin-top: 8px;
  font-weight: 500;
}

.styled-table {
  width: 100%;
  border-collapse: collapse;
  margin: 16px 0;
  font-size: 0.9em;
  page-break-inside: avoid;
}
.styled-table th, .styled-table td {
  border: 1px solid #cbd5e1;
  padding: 8px 10px;
  text-align: left;
}
.styled-table th {
  background: #f1f5f9;
  font-weight: 700;
  color: #1e293b;
}

.code-block {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 5px;
  padding: 10px 12px;
  font-family: 'Courier New', monospace;
  font-size: 0.85em;
  overflow: visible;
  page-break-inside: avoid;
}

ul { padding-left: 20px; margin: 6px 0; }
li { margin-bottom: 4px; }
p { margin: 6px 0; }
hr { border: 0; border-top: 1px dashed #cbd5e1; margin: 25px 0; }
</style>
</head>
<body>

<div class="cover-page-wrapper">
  <img src="assets/cover.jpg" class="cover-img" alt="Cover" />
</div>

<div class="book-container">
${bodyHtml}
</div>

</body>
</html>`;
}

fs.writeFileSync(path.join(baseDir, 'index.html'), createHtmlDocument(false), 'utf8');
fs.writeFileSync(path.join(baseDir, 'index_kindle.html'), createHtmlDocument(true), 'utf8');
console.log('build_perfect_pdf.js successfully compiled index.html and index_kindle.html with 100% KaTeX pre-rendering!');
