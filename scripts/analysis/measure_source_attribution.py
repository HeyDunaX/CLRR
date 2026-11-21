"""Exact coupled-mask frozen SAC probe; no optimizer or test access.

Uses the pinned restore, RNG replay, resource admission and cleanup path of
measure_source_response. Cell order is [mask, source], not [source, mask].
"""
import argparse
import math
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
import torch.nn.functional as F
from methods.source_attribution.v1.information import information_components


def sac_metrics(a,b,u,v,labels,tail=None):
    table=torch.stack((torch.stack((a,u),-2),torch.stack((b,v),-2)),-3)
    info=information_components(table)
    assert (info['source_attribution']>=-2e-6).all()
    assert (info['source_attribution']<=info['joint_dropout']+2e-6).all()
    assert (info['joint_dropout']<=math.log(2)+2e-6).all()
    assert (info['joint_dropout']-info['marginal_dropout']-info['source_attribution']).abs().max()<2e-6
    logs=[F.log_softmax(t.float(),-1) for t in (a,b,u,v)]
    la,lb,lu,lv=logs
    y=labels[:,None]
    gaps=[(la-lu).gather(1,y).squeeze(1),(lb-lv).gather(1,y).squeeze(1)]
    matched=.5*(la.exp()+lb.exp())
    entropy=-.5*((la.exp()*la).sum(-1)+(lb.exp()*lb).sum(-1))
    return dict(C=info['source_attribution'],A=info['marginal_dropout'],
        B=info['joint_dropout'],source_information=info['source_information'],
        chain_residual=(info['joint_dropout']-info['marginal_dropout']-info['source_attribution']).abs(),
        source_gold_gap=.5*(gaps[0]+gaps[1]),source_gap_mask_difference=(gaps[0]-gaps[1]).abs(),
        ce=-.5*(la.gather(1,y)+lb.gather(1,y)).squeeze(1),
        entropy=entropy,gold_probability=matched.gather(1,y).squeeze(1),
        top1_error=1-.5*(a.argmax(-1).eq(labels).float()+b.argmax(-1).eq(labels).float()))


def cpu_gate():
    g=torch.Generator().manual_seed(42)
    a,b,u,v=[torch.randn(7,19,generator=g) for _ in range(4)]
    y=torch.arange(7)
    whole=sac_metrics(a,b,u,v,y)
    for k,t in whole.items():
        assert torch.isfinite(t).all(),k
        chunks=torch.cat([sac_metrics(a[i:i+1],b[i:i+1],u[i:i+1],v[i:i+1],y[i:i+1])[k] for i in range(7)])
        torch.testing.assert_close(t,chunks)
    torch.testing.assert_close(whole['chain_residual'],torch.zeros(7),atol=2e-6,rtol=0)
    blind=sac_metrics(a,b,a,b,y)
    stable=sac_metrics(a,a,u,u,y)
    for result in (blind,stable):
        torch.testing.assert_close(result['C'],torch.zeros(7),atol=2e-6,rtol=0)
    # Independent direct chain identity catches axis-order mistakes.
    def js(x,z):
        lx,lz=x.log_softmax(-1),z.log_softmax(-1)
        mixture=torch.logaddexp(lx,lz)-math.log(2)
        return .5*((lx.exp()*(lx-mixture)).sum(-1)+(lz.exp()*(lz-mixture)).sum(-1))
    torch.testing.assert_close(whole['B'],.5*(js(a,b)+js(u,v)),atol=2e-6,rtol=1e-5)
    from measure_source_response import cpu_gate as native_cpu_gate
    native=native_cpu_gate()
    return dict(passed=True,checks=['cell_axis_order','chunk_replay','chain_identity','source_blind_zero','stable_source_zero'],native=native,optimizer_steps=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--plan');p.add_argument('--output');p.add_argument('--cpu-gate',action='store_true')
    args=p.parse_args()
    if args.cpu_gate: print(json.dumps(cpu_gate()))
    else:
        assert args.plan and args.output
        assert json.loads(Path(args.plan).read_text())['metric_family']=='source_attribution_v1'
        from measure_source_response import run
        run(args)
