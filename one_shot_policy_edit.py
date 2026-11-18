"""Evaluate a fixed game-aware edit with independent-seed, fixed-cohort episodes.

No calibration, fitting, candidate search or private answer key. The edit rule is
specified independently of held-out scores. See report for exact update/hash.
"""
from __future__ import annotations
import argparse, copy, hashlib, io, json, platform
from pathlib import Path
import numpy as np
import torch
from games import GridGame
from train import Policy
from benchmark_policy_efficiency import latency
from final_evaluate import CHECKPOINTS


def cohort(model, seed, episodes):
    env = GridGame('pong', episodes, seed)
    active = np.ones(episodes, bool)
    returns = np.zeros(episodes)
    for _ in range(150):
        action = model(torch.from_numpy(env.observe().astype(np.float32))).argmax(-1).numpy()
        _, reward, done = env.step(action)
        returns[active] += reward[active]
        active &= ~done
        if not active.any(): break
    return returns


def summarize(values):
    a = np.array(values)
    return {'episodes': len(a), 'mean_return': float(a.mean()),
            'se_return': float(a.std(ddof=1)/np.sqrt(len(a))),
            'min_return': float(a.min()), 'max_return': float(a.max())}


def proposed_edit(state, revision=False):
    original_head = state['layers.4.weight'].clone()
    bound = float((original_head.abs().sum(1) + state['layers.4.bias'].abs()).max())
    state['layers.0.weight'][0].zero_()
    for row in range(10):
        state['layers.0.weight'][0, row*10] = -row
        state['layers.0.weight'][0, 100+row*10:100+(row+1)*10] = row
    state['layers.0.bias'][0] = 0
    state['layers.2.weight'][0].zero_()
    state['layers.2.weight'][0, 0] = 1
    state['layers.2.bias'][0] = 0
    gain = (8 if revision else 4)*(1+bound)
    if revision:
        state['layers.4.bias'][0] += 2*bound+1
    state['layers.4.weight'][1, 0] -= gain
    state['layers.4.weight'][2, 0] += gain
    return state, {'author': 'Astra design agent, fixed before evaluation',
                   'kind': 'one composite analytic intervention across three layers',
                   'rule': 'Repurpose first hidden unit to tanh(ball row minus paddle row); relay via tanh; add opposite up/down steering head coefficients.',
                   'original_logit_bound': bound, 'head_gain': gain,
                   'calibration_episodes': 0, 'candidate_search': False,
                   'revision': 2 if revision else 1,
                   'revision_note': 'Follow-up after failed first proposal: bias toward stay at equal rows and stronger directional steering.' if revision else 'Original frozen proposal.',
                   'limitation': 'Embeds a hand-specified Pong tracking heuristic; multiple parameter coordinates change. This is not a discovered task-agnostic latent edit.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/one_shot_policy_edit.json')
    parser.add_argument('--episodes-per-seed',type=int,default=512)
    parser.add_argument('--seeds',type=int,nargs='+',default=[40103,50627,60733,70841,80953])
    parser.add_argument('--revision',action='store_true',help='Explicitly label the second attempt after the original failed.')
    parser.add_argument('--random-controls',type=int,default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint = CHECKPOINTS['pong']
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)['policy']
    edited, description = proposed_edit(copy.deepcopy(state), args.revision)
    delta = {k: edited[k]-state[k] for k in state}
    variants={'baseline':state, 'astra_fixed_edit':edited}
    # Same support and tensorwise Frobenius magnitude as the proposed intervention.
    for seed in range(args.random_controls):
        rng=torch.Generator().manual_seed(301+seed)
        random_state=copy.deepcopy(state)
        for k,d in delta.items():
            mask=d!=0
            if mask.any():
                noise=torch.randn(d.shape,generator=rng)*mask
                random_state[k]+=noise*(d.norm()/noise.norm())
        variants[f'random_edit_{seed}']=random_state
    report={'config':vars(args),'game':'pong','checkpoint':str(checkpoint),
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'proposal':description,'environment':{'python':platform.python_version(),'torch':torch.__version__, 'threads':1},
            'changed_scalars':sum(int((d!=0).sum()) for d in delta.values()),
            'notes':['Fixed analytic game-aware heuristic; no training-seed selection or calibration.',
                     'Each seed uses one fixed cohort; terminal episodes are counted once, without resetting or completion-order selection.',
                     'Random edits match changed-coordinate support and per-tensor update norm; they are not generic random policies.',
                     'Latency is warm CPU inference only. Timing differences are noisy; architecture and parameter count are identical.',
                     'Reported standard errors describe episodes, not independent seed-level uncertainty.'], 'variants':{}}
    with torch.inference_mode():
        for name,s in variants.items():
            model=Policy().eval();model.load_state_dict(s)
            per_seed={str(seed):cohort(model,seed,args.episodes_per_seed).tolist() for seed in args.seeds}
            all_values=sum(per_seed.values(),[])
            buffer=io.BytesIO();np.savez(buffer,**{k:v.numpy() for k,v in s.items()})
            report['variants'][name]={'evaluation':summarize(all_values),'per_seed':{k:summarize(v) for k,v in per_seed.items()},'episode_returns':per_seed,
                'raw_tensor_bytes':sum(v.numel()*v.element_size() for v in s.values()),'npz_bytes':len(buffer.getvalue()),
                'state_sha256':hashlib.sha256(buffer.getvalue()).hexdigest(),
                'latency_batch1':latency(model,torch.zeros(1,500),1000,9)}
            print(name,report['variants'][name]['evaluation'],flush=True)
    base=report['variants']['baseline']['episode_returns']
    for name,v in report['variants'].items():
        diffs=np.concatenate([np.array(v['episode_returns'][str(seed)])-base[str(seed)] for seed in args.seeds])
        v['paired_return_change']=summarize(diffs)
        v['paired_fraction_improved']=float((diffs>0).mean())
    output=Path(args.output);output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    torch.save({'policy':edited,'edit_description':description},output.with_suffix('.pt'))

if __name__=='__main__':main()
