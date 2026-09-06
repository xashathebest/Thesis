"""Rebuild the evidence-only comparison notebook without embedded outputs."""
import json
from pathlib import Path

def cell(kind, source):
    value = dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True))
    if kind=='code':
        value.update(execution_count=None,outputs=[])
    return value

cells = [cell('markdown', '''# Model comparison
Actual results only. Box metrics compare the same 12-class part target; mask metrics are additional YOLO capability. Missing results remain unavailable. Read results/MODEL_COMPARISON.md for protocol, provenance and limitations.
'''),cell('code','''from pathlib import Path
import csv
import html
try:
    from IPython.display import HTML, display
except ImportError:
    HTML = str
    display = print
import matplotlib.pyplot as plt
root = Path.cwd() if (Path.cwd() / 'results').exists() else Path.cwd().parent
def read_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))
def show_table(rows):
    if not rows:
        print('No measured results')
        return
    fields = list(rows[0])
    header = '<tr>' + ''.join('<th>'+html.escape(k)+'</th>' for k in fields) + '</tr>'
    body = ''.join('<tr>'+''.join('<td>'+html.escape(str(row.get(k) if row.get(k) not in ('', None) else 'N/A'))+'</td>' for k in fields)+'</tr>' for row in rows)
    display(HTML('<table>'+header+body+'</table>'))
rows = read_rows(root / 'results/model_comparison.csv')
show_table(rows)
measured = [row for row in rows if row['status'] == 'completed']
'''),cell('code','''columns = ['mAP50', 'mAP50_95', 'Precision', 'Recall', 'FPS', 'ModelSize']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, metric in zip(axes.flat, columns):
    available = [row for row in measured if row[metric] not in ('', 'N/A')]
    if available:
        ax.bar([row['Model'] for row in available], [float(row[metric]) for row in available])
        ax.tick_params(axis='x', rotation=25)
    else:
        ax.text(.5, .5, 'No measured results', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(metric + (' (MB)' if metric == 'ModelSize' else ''))
    if metric in columns[:4]:
        ax.set_ylim(0, 1)
fig.tight_layout()
plt.show()
'''),cell('markdown','''## Per-class and grouped evidence
Macro quality/anatomy means retain the original part semantics. Low minority-class recall matters for conveyor decisions; these groups are not whole-fish detections.
'''),cell('code','''import json
for family in ['yolo', 'faster_rcnn', 'efficientdet_d3']:
    print(family)
    path = root / 'results' / family
    if (path / 'per_class.csv').exists():
        show_table(read_rows(path / 'per_class.csv'))
    if (path / 'groups.json').exists():
        groups = json.loads((path / 'groups.json').read_text())
        show_table([{'group':group, **values} for group, values in groups.items()])
'''),cell('markdown','''## Additional mask capability
YOLO native mask P/R use the native operating point. They are separate from the common box measurements and do not penalize detection-only models.
'''),cell('code','''show_table([{k:row[k] for k in ['Model', 'mask_precision', 'mask_recall', 'mask_map50', 'mask_map50_95']} for row in rows])
print('No automatic conveyor winner: require measured accuracy, class recall, speed, size and the application frame budget.')
''')]
for index,item in enumerate(cells):
    item['id'] = f'comparison-{index}'
Path(__file__).with_name('03_model_comparison.ipynb').write_text(json.dumps(dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),nbformat=4,nbformat_minor=5),indent=1)+'\n',encoding='utf-8')
