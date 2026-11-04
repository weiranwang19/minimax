"""Check production NCWC against saved direct-certificate experiments.

Writes only implementation_replays/, preserving the original experiment
artifacts. No W&B runs are created. Each accepted endpoint is independently
verified by differentiating the original scalar objective with autograd.
"""
import argparse
import json
import math
from pathlib import Path
import signal
import time

import torch

import experiment as saved


OUT = Path(__file__).resolve().parent


class Audit:
    def __init__(self):
        self.rule = 'direct'
        self.calls = []

    def run(self, *args, **kwargs):
        mins, maxs, h = args[:3]
        bounds = saved.mm._normalize_box_bounds(kwargs['box_bounds'], len(mins), len(maxs))
        call = {'iterations': [], 'complete': False}
        self.calls.append(call)
        previous_callback = kwargs.get('progress_callback')

        def progress(p):
            if p['completed_outer_iters']:
                assert p['inner_terminated']
                assert p['scsc_final_delta'] <= p['epsilon_k']
                row = dict(iteration=p['completed_outer_iters'], epsilon=p['epsilon'],
                           tau=p['epsilon_k'], step=p['final_diff'],
                           r_min=p['final_residual_min'], r_max=p['final_residual_max'],
                           scsc_inner=p['scsc_num_inner_iters'], scsc_outer=p['scsc_num_outer_iters'],
                           stop=p['ncwc_terminated'], termination_reason=p['termination_reason'])
                call['iterations'].append(row)
                if row['stop']:
                    ru, rv = saved.autograd_kkt(h, mins, maxs, bounds)
                    assert max(ru, rv) <= p['epsilon']
                    assert math.isclose(ru, row['r_min'], abs_tol=1e-12, rel_tol=1e-12)
                    assert math.isclose(rv, row['r_max'], abs_tol=1e-12, rel_tol=1e-12)
                    assert p['termination_reason'] == 'direct_certificate'
                    row.update(autograd_min=ru, autograd_max=rv)
            if previous_callback is not None:
                previous_callback(p)

        result = saved.mm.optimize_NCWC(*args, **{**kwargs, 'progress_callback': progress})
        call['complete'] = result['terminated']
        assert result['terminated'], result
        return result

    def summary(self):
        return saved.Audit.summary(self)


def compare(audit, result, reference, analytic=False):
    assert audit.summary() == reference['summary']
    if analytic:
        for key in ('u', 'v', 'objective'):
            assert math.isclose(result[key], reference['result'][key], abs_tol=1e-12)
    else:
        for current, old in zip(audit.calls, reference['calls']):
            assert len(current['iterations']) == len(old['iterations'])
            for a, b in zip(current['iterations'], old['iterations']):
                for key in ('iteration', 'scsc_inner', 'scsc_outer', 'stop'):
                    assert a[key] == b[key], (key, a[key], b[key])
                for key in ('step', 'r_min', 'r_max'):
                    assert math.isclose(a[key], b[key], abs_tol=1e-12, rel_tol=1e-12), key
        for key in ('x', 'y', 'z', 'multipliers'):
            torch.testing.assert_close(torch.tensor(result[key]), torch.tensor(reference['result'][key]),
                                       atol=1e-12, rtol=1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=('analytic', 'smo_stage6', 'smo_stage8', 'smo_small_stage14'))
    parser.add_argument('--seconds', type=int, default=180)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    def timeout(*_):
        raise TimeoutError(f'{args.seconds} second replay budget exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    start = time.monotonic()
    output = {'case': args.case, 'status': 'complete', 'comparisons': []}
    try:
        if args.case == 'analytic':
            references = [r for r in json.loads((OUT/'analytic.json').read_text()) if r['rule'] == 'direct']
            for reference in references:
                audit = Audit()
                result = saved.analytic(audit, reference['case'])
                compare(audit, result, reference, analytic=True)
                output['comparisons'].append(dict(case=reference['case'], summary=audit.summary(),
                                                  result=result, calls=audit.calls))
        else:
            size, epsilon = {'smo_stage6': ((100, 100, 5), .3),
                             'smo_stage8': ((100, 100, 5), .2),
                             'smo_small_stage14': ((4, 4, 2), .05)}[args.case]
            filename = 'smo-'+'-'.join(map(str, size))+f'-idx1-eps{epsilon}-direct.json'
            reference = json.loads((OUT/filename).read_text())
            audit, stages = Audit(), []
            result = saved.smo(audit, size, 1, epsilon, stages)
            compare(audit, result, reference)
            output['comparisons'].append(dict(reference=filename, summary=audit.summary(),
                                              result=result, calls=audit.calls, stages=stages))
    except TimeoutError as error:
        output.update(status='incomplete', error=str(error))
    finally:
        signal.alarm(0)
    output['seconds'] = time.monotonic()-start
    target = OUT/'implementation_replays'
    target.mkdir(exist_ok=True)
    (target/f'{args.case}.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'comparisons'}), flush=True)


if __name__ == '__main__':
    main()
