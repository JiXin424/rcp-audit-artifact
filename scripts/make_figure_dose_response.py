#!/usr/bin/env python3
"""Generate the dose-response figure (Fig. 3) from results/dose_response.json."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = json.load(open(ROOT / 'results/dose_response.json'))
points = d['points']

# Filter out None values
pts = [p for p in points if p.get('dev_nll') is not None and p.get('gap') is not None]

# Family colors
COLORS = {
    'released': 'red',
    'reconstructions_primary': 'C0',
    'reconstructions_extension': 'C0',
    'distillation': 'C1',
}
MARKERS = {
    'released': '*',
    'reconstructions_primary': 'o',
    'reconstructions_extension': 'o',
    'distillation': 's',
}

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each family
for fam in ['reconstructions_primary', 'reconstructions_extension', 'distillation', 'released']:
    fam_pts = [p for p in pts if p['family'] == fam]
    if not fam_pts:
        continue
    xs = [p['dev_nll'] for p in fam_pts]
    ys = [p['gap'] for p in fam_pts]
    sizes = [200 if fam == 'released' else 50 for _ in fam_pts]
    ax.scatter(xs, ys, c=COLORS[fam], marker=MARKERS[fam], s=sizes,
               label=fam.replace('_', ' ').title(), alpha=0.7,
               edgecolors='black' if fam == 'released' else None,
               linewidths=1.5 if fam == 'released' else 0)

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Dev NLL (per token; lower = more competent)', fontsize=12)
ax.set_ylabel('PHX-public PURE--GT gap (BLEU points)', fontsize=12)
ax.set_title('Dose-response: BT-evaluator competence vs retrieval stress-test gap\n'
             '(Greedy decode; 14 reconstructions + 15 distillation + released; 43 total beam-3 in text)', fontsize=11)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

# Annotate the released star
released_pt = [p for p in pts if p['family'] == 'released'][0]
ax.annotate(f"Released\ngap={released_pt['gap']:+.2f}",
            xy=(released_pt['dev_nll'], released_pt['gap']),
            xytext=(released_pt['dev_nll'] + 0.15, released_pt['gap'] - 1),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
out_pdf = ROOT / 'generated_figures/dose_response.pdf'
out_png = ROOT / 'generated_figures/dose_response.png'
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_png, bbox_inches='tight', dpi=150)
print(f'Wrote: {out_pdf}')
print(f'Wrote: {out_png}')

# Stats summary
gaps = [p['gap'] for p in pts if p['family'] != 'released']
print(f'\nFigure stats:')
print(f'  {len(gaps)} trained checkpoints')
print(f'  Gap range: [{min(gaps):.2f}, {max(gaps):.2f}]')
print(f'  Released gap: {released_pt["gap"]:+.2f} at dev_nll {released_pt["dev_nll"]:.2f}')
