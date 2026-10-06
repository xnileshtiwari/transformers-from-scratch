with open('/home/nilesh/code/transformers-from-scratch/Transformers_From_Scratch.md', 'r', encoding='utf-8') as f:
    text = f.read()

import re
# Find all ``` blocks
blocks = re.findall(r'```(\w*)\n([\s\S]*?)```', text)
print(f"Total code blocks in markdown: {len(blocks)}")

ascii_blocks = []
for lang, content in blocks:
    if lang in ['mermaid', 'xml', 'svg']:
        continue
    # Check if content has box-drawing characters or ascii art
    if any(c in content for c in ['┌', '─', '│', '└', '▼', '►', '+--', '|', '-->']):
        ascii_blocks.append((lang, content))

print(f"Remaining ASCII diagrams: {len(ascii_blocks)}")
for i, (lang, content) in enumerate(ascii_blocks):
    first_few = [l for l in content.split('\n') if l.strip()][:3]
    print(f"[{i+1}] (lang='{lang}'):", ' | '.join(first_few)[:80])
