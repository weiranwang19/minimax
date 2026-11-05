"""Plot frozen-input certificate checks and reference NCWC continuation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
trace=json.loads((OUT/'trace100.json').read_text())
continuation=json.loads((OUT/'reference_ncwc.json').read_text())
rows=trace['production_100']['trace']
fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
ax=axes[0]
ax.semilogy([r['outer'] for r in rows],[r['projected_residual'] for r in rows],
            color='#4267ac',label='Projected certificate')
ax.semilogy([r['outer'] for r in rows],[r['feasible_kkt']['joint'] for r in rows],
            color='#bb6338',label='Independent KKT at feasible output')
ax.axhline(trace['params']['tau'],color='#333',linestyle='--',label='Required inner tolerance')
ax.set(xlabel='SCSC outer iteration',ylabel='Residual norm',title='Same captured FOP subproblem')
ax.legend(fontsize=8);ax.grid(alpha=.15)
ax=axes[1];rows=continuation['rows']
for block,color in [('min','#bb6338'),('max','#16806a')]:
    ax.semilogy([r['iteration'] for r in rows],[r['original_kkt'][block] for r in rows],
                color=color,label=f'Original-stage {block} KKT residual')
ax.axhline(continuation['epsilon'],color='#333',linestyle='--',label='Required NCWC tolerance')
ax.set(xlabel='NCWC iteration',ylabel='Residual norm',title='NCWC with certified reference subsolves')
ax.legend(fontsize=8);ax.grid(alpha=.15)
for ext in ('png','pdf'):
    fig.savefig(OUT/f'comparison.{ext}',dpi=170)
