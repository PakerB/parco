import json

with open('examples/4.quickstart-pvrpwdp.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Number of cells: {len(data['cells'])}")
print("\nFirst 10 cell types:")
for i, cell in enumerate(data['cells'][:10]):
    print(f"  {i+1}. {cell['cell_type']}")

print("\nAll cell types:")
for i, cell in enumerate(data['cells']):
    source_preview = ''.join(cell['source'][:50]) if cell['source'] else ''
    print(f"  {i+1}. {cell['cell_type']:10s} - {source_preview[:40]}...")
