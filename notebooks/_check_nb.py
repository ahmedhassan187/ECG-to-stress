"""Check the FFT notebook."""
import json

NB_PATH = r'g:\Master\Thesis\FLT\Code\ECG-to-stress\notebooks\06_FFT_analysis.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

result = []
result.append('nbformat: ' + str(nb.get('nbformat')))
result.append('cells: ' + str(len(nb.get('cells', []))))
for i, c in enumerate(nb.get('cells', [])):
    src = c.get('source', [])
    src_len = len(src) if isinstance(src, list) else 1
    result.append(f'  cell {i}: id={c.get("id","?")}, source_lines={src_len}')

# Write to a file
with open(NB_PATH.replace('.ipynb', '_info.txt'), 'w') as f:
    f.write('\n'.join(result))
print('Done')
