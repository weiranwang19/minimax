"""Read the latest four local W&B histories and audit the stopping tests.

No network access, solver execution, or experiment-setting changes. History
files are opened read-only. Only this report's results.json is written.
"""
from pathlib import Path
import json
import math
import re
import sys

import torch
import yaml
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import exp_cblo as experiment


def history(run):
    path = next(run.glob('*.wandb'))
    store = DataStore()
    store._fp = path.open('rb')
    store._size_bytes = path.stat().st_size
    store._opened_for_scan = True
    store._read_header()
    rows = []
    try:
        while (data := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.HasField('history'):
                rows.append({item.key or '/'.join(item.nested_key): json.loads(item.value_json)
                             for item in record.history.item})
    finally:
        store._fp.close()
    return rows


def analyze(run):
    config = {k: v['value'] for k, v in yaml.safe_load((run/'files/config.yaml').read_text()).items()
              if isinstance(v, dict) and 'value' in v and not k.startswith('_')}
    rows = history(run)
    ncwc = [r for r in rows if 'ncwc/completed_outer_iters' in r]
    last = ncwc[-1]
    call = last['ncwc/call_index']
    active = [r for r in ncwc if r['ncwc/call_index'] == call and r['ncwc/completed_outer_iters'] > 0]
    method, size = config['solver_method'], config['problem_size']
    experiment.generate_instance((size['n'], size['m'], size['l']), config['instance_idx'])
    lg = experiment.CURRENT_L_G
    if method == 'fop':
        stage = call-1
        rho = config['base_rho'] ** (stage-1)
        epsilon, mu = 1/rho, rho**2
        lip = 4*rho*mu*lg**2
        tau0 = epsilon**2.5
        diameter = 2*math.sqrt(size['m'])
    elif method == 'smo':
        stage = call-1
        epsilon = config['smo_epsilon_0'] * config['smo_tau']**stage
        rho, mu = epsilon**-1, epsilon**-3
        lip, tau0 = 2*mu*lg**2, .5*epsilon**2.5
        diameter = 2*math.sqrt(size['m'])
    else:
        stage = 0
        epsilon = config['gcmo_eps']
        rho, mu = epsilon**-1, None
        lip, tau0 = 2*rho*lg, epsilon**1.5
        diameter = math.sqrt(4*size['m'] + config['gcmo_lagrange_bound']**2*size['l'])
    assert config.get('fop_lip_scale') is None and config.get('smo_lip_scale') is None
    assert config.get('gcmo_lip_override') is None
    completed = last['ncwc/completed_outer_iters']
    step = last['ncwc/final_diff']
    tau = tau0/completed
    target = epsilon/(4*lip)
    bounds = (tau+2*lip*step, tau+epsilon/2)
    first_blockwise = first_joint = None
    for r in active:
        t = tau0/r['ncwc/completed_outer_iters']
        a, b = t+2*lip*r['ncwc/final_diff'], t+epsilon/2
        brief = dict(completed=r['ncwc/completed_outer_iters'], runtime=r['_runtime'],
                     min_bound=a, max_bound=b, joint_bound=math.hypot(a, b),
                     subsequent_runtime=last['_runtime']-r['_runtime'])
        if first_blockwise is None and max(a, b) <= epsilon:
            first_blockwise = brief
        if first_joint is None and math.hypot(a, b) <= epsilon:
            first_joint = brief
    lines = (run/'files/output.log').read_text().splitlines()
    outer = [(i+1, s) for i, s in enumerate(lines) if s.startswith('optimize_NCWC outer=')]
    scsc = [(i+1, s) for i, s in enumerate(lines) if s.startswith('Minimax_SCSC outer=')]
    certified = [row for row in scsc if 'termination_reason=inner_certificate' in row[1]]
    window = active[-100:]
    last_window = {}
    for field in ('upper_obj', 'final_diff', 'feas', 'lower_gap'):
        values = [r['ncwc/'+field] for r in window]
        last_window[field] = dict(first=values[0], last=values[-1], min=min(values), max=max(values))
    avg_change = ((window[-1]['ncwc/upper_obj']-window[0]['ncwc/upper_obj'])
                  /(window[-1]['ncwc/completed_outer_iters']-window[0]['ncwc/completed_outer_iters']))
    return dict(run=run.name, method=method, size=size, active_stage=stage, active_ncwc_call=call,
                runtime=last['_runtime'], epsilon=epsilon, rho=rho, mu=mu, lip=lip,
                completed=completed, tau=tau, step=step, step_target=target, step_ratio=step/target,
                condition_bound=(3*lip+epsilon/(2*diameter))/(epsilon/(2*diameter)),
                original_stationarity_upper_bounds=dict(min=bounds[0], max=bounds[1], joint=math.hypot(*bounds)),
                first_blockwise_bound_success=first_blockwise, first_joint_bound_success=first_joint,
                total_completed_scsc=len(outer), inner_certificate_returns=len(certified),
                failed_scsc=sum('inner_terminated=False' in row[1] for row in outer),
                inner_progress_lines=sum(s.startswith('Minimax_SCSC inner ') for s in lines),
                interrupted=any('KeyboardInterrupt' in s for s in lines),
                last_certificate=certified[-1], last_ncwc=outer[-1], last_scsc=scsc[-1],
                last_metrics={k: v for k, v in last.items() if k.startswith('ncwc/')},
                last100=last_window, mean_objective_change_per_iteration=avg_change,
                linear_face_descent_prediction=-2*lip*step**2)


if __name__ == '__main__':
    torch.set_num_threads(1)
    results = [analyze(run) for run in sorted((ROOT/'wandb').glob('run-*'), reverse=True)[:4]]
    (Path(__file__).resolve().parent/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    for r in results:
        print(r['run'], 'step/target=', r['step_ratio'], 'tau=', r['tau'],
              'residual bound=', r['original_stationarity_upper_bounds'],
              'failed SCSC=', r['failed_scsc'])
