"""Bounded continuation of the captured NCWC call with reference subsolves."""
import json
import signal
import time

import torch

import diagnose as d


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    data = torch.load(d.OUT/'fop_stage4.pt',weights_only=False)
    epsilon=.008; tau0=data['params']['tau']
    rows=[];start=time.monotonic()
    def timeout(*_):
        raise TimeoutError('Reference continuation exceeded 120 seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    output={'status':'max_iter','epsilon':epsilon}
    try:
        for j in range(1,1001):
            data['params']['tau']=tau0/j
            result=d.reference(data)
            row=dict(iteration=j,tau=data['params']['tau'],regularized_kkt=result['kkt'],
                     original_kkt=result['original_kkt'])
            rows.append(row)
            if not result['passes']:
                output['status']='uncertified_reference'
                break
            if max(result['original_kkt']['min'],result['original_kkt']['max'])<=epsilon:
                output['status']='converged'
                break
            data['mins']=[torch.tensor(v) for v in result['mins']]
            data['warm_dual']=result['dual_weights']
            if j%100==0:
                print(json.dumps(row),flush=True)
    except TimeoutError as error:
        output.update(status='timeout',error=str(error))
    finally:
        signal.alarm(0)
    output.update(seconds=time.monotonic()-start,iterations=len(rows),rows=rows)
    (d.OUT/'reference_ncwc.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    main()
