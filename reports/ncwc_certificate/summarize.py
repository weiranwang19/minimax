"""Summarize the actual experiments and the historical SMO stopping bound."""
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT/'reports/latest_four'))


def main():
    from analyze import history
    experiments = []
    for path in sorted(OUT.glob('smo-*.json')):
        data = json.loads(path.read_text())
        last = data['stages'][-1] if data['stages'] else None
        endpoints = [c['iterations'][-1] for c in data['calls'] if c['complete']]
        experiments.append(dict(file=path.name, config=data['config'], status=data['status'],
                                error=data.get('error'), summary=data['summary'], seconds=data['seconds'],
                                last_stage=last,
                                worst_accepted_block_ratio=max(max(r['autograd_min'],r['autograd_max'])/r['epsilon'] for r in endpoints),
                                worst_accepted_joint_ratio=max(r['r_joint']/r['epsilon'] for r in endpoints)))

    run = ROOT/'wandb/run-20261023_131037-22bvabs6'
    all_history = history(run)
    rows = [r for r in all_history if r.get('ncwc/call_index') == 15
            and (r.get('ncwc/completed_outer_iters') or 0) > 0]
    audited = next(r for r in json.loads((ROOT/'reports/latest_four/results.json').read_text())
                   if r['method'] == 'smo')
    epsilon, lip = audited['epsilon'], audited['lip']
    k = np.array([r['ncwc/completed_outer_iters'] for r in rows])
    step = np.array([r['ncwc/final_diff'] for r in rows])
    tau = .5*epsilon**2.5/k
    min_bound, max_bound = tau+2*lip*step, tau+epsilon/2
    first = int(np.flatnonzero(np.maximum(min_bound,max_bound) <= epsilon)[0])
    historical = dict(run=run.name, stage=14, epsilon=epsilon, lip=lip,
                      first_bound_iteration=int(k[first]),
                      later_iterations=int(k[-1]-k[first]),
                      later_recorded_hours=(rows[-1]['_runtime']-rows[first]['_runtime'])/3600,
                      final_step=step[-1], old_threshold=epsilon/(4*lip),
                      improved_threshold=(epsilon-tau[-1])/(2*lip),
                      final_min_bound=min_bound[-1], final_max_bound=max_bound[-1])
    replay = json.loads((OUT/'smo-100-100-5-idx1-eps0.2-baseline.json').read_text())
    matched = 0
    for call, data in enumerate(replay['calls'], 1):
        old = [r for r in all_history if r.get('ncwc/call_index') == call
               and (r.get('ncwc/completed_outer_iters') or 0) > 0]
        assert len(old) >= len(data['iterations'])
        for a, b in zip(data['iterations'], old):
            assert a['iteration'] == b['ncwc/completed_outer_iters']
            assert a['step'] == b['ncwc/final_diff']
            assert a['scsc_inner'] == b['ncwc/scsc_inner_iters']
            matched += 1
    assert matched == replay['summary']['ncwc_iterations']
    historical_stages = [[r for r in all_history if r.get('ncwc/call_index') == call][-1]
                         for call in range(1, 10)]
    baseline_reference = dict(source=run.name, matched_replay_iterations=matched,
                              ncwc_iterations=sum(r['ncwc/completed_outer_iters'] for r in historical_stages),
                              scsc_inner=historical_stages[-1]['ncwc/cumulative_scsc_inner_iters'],
                              final_metrics=historical_stages[-1])
    verification = dict(
        accepted_smo_stages=sum(r['summary']['completed_calls'] for r in experiments),
        worst_accepted_block_ratio=max(r['worst_accepted_block_ratio'] for r in experiments),
        worst_accepted_joint_ratio=max(r['worst_accepted_joint_ratio'] for r in experiments),
    )
    (OUT/'summary.json').write_text(json.dumps(dict(verification=verification, historical=historical,
        stage8_baseline_reference=baseline_reference, experiments=experiments),indent=2)+'\n')
    (OUT/'plot_data.json').write_text(json.dumps(dict(k=k.tolist(), step=step.tolist(),
        min_bound=min_bound.tolist(), max_bound=max_bound.tolist(), first=first,
        epsilon=epsilon, lip=lip, historical_stage_work=[r['ncwc/cumulative_scsc_inner_iters']
                                                       for r in historical_stages]))+'\n')
    for row in experiments:
        print(row['file'],row['status'],row['summary'],
              'max accepted residual/epsilon',row['worst_accepted_block_ratio'])


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads((OUT/'plot_data.json').read_text())
    k, step, min_bound, max_bound = (np.array(data[key]) for key in ('k','step','min_bound','max_bound'))
    epsilon, lip, first = data['epsilon'], data['lip'], data['first']
    historical_stage_work = data['historical_stage_work']

    fig, axes = plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    ax = axes[0]
    ax.plot(k,step/(epsilon/(4*lip)),color='#4267ac',label='Existing step test')
    ax.plot(k,np.maximum(min_bound,max_bound)/epsilon,color='#16806a',label='Certified residual bound')
    ax.axhline(1,color='#333333',linewidth=1,linestyle='--')
    ax.axvline(k[first],color='#777777',linewidth=1,linestyle=':')
    ax.text(k[first]+80,1.9,f'Bound passes at {k[first]}',fontsize=9)
    ax.set(yscale='log',ylim=(.4,12),xlabel='NCWC iteration within SMO stage 14',
           ylabel='Test value / allowed maximum',title='Historical 100-dimensional run')
    ax.legend(fontsize=9)
    ax.grid(alpha=.15)

    ax = axes[1]
    colors={'baseline':'#4267ac','bound':'#16806a','direct':'#bb6338'}
    labels={'baseline':'Existing step test','bound':'Residual bound','direct':'Direct KKT'}
    for rule in ('baseline','bound','direct'):
        path=OUT/f'smo-100-100-5-idx1-eps0.2-{rule}.json'
        if not path.exists():
            path=OUT/f'smo-100-100-5-idx1-eps0.3-{rule}.json'
        data=json.loads(path.read_text())
        ax.plot([s['stage'] for s in data['stages']],np.cumsum([s['scsc_inner'] for s in data['stages']]),
                marker='o',markersize=4,color=colors[rule],label=labels[rule])
        if rule == 'baseline' and not data['status'] == 'complete':
            ax.plot([7,8],historical_stage_work[7:9],
                    linestyle='--',color=colors[rule],marker='D',markerfacecolor='white',
                    label='Recorded baseline continuation')
    ax.set(xlabel='Completed SMO stage (zero based)',ylabel='Cumulative SCSC inner updates',
           title='New runs: changed subsequent trajectories')
    ax.legend(fontsize=9)
    ax.grid(alpha=.15)
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'comparison.{ext}',dpi=170)


if __name__ == '__main__':
    plot() if '--plot' in sys.argv else main()
