"""Reproduce CPU policy size/quality/latency measurements; no blind keys needed.

Int8 storage variants dequantize once before inference. dynamic_int8 uses native
quantized Linear with dynamic activation quantization, when the backend permits.
Bytes are uncompressed NPZ payloads, except native PyTorch serialized state.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import platform
import time
import warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from final_evaluate import CHECKPOINTS
from games import GridGame
from train import Policy, evaluate, scripted_actions


def artifact_size(arrays):
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return len(stream.getvalue())


def storage_variant(state, scheme):
    stored, restored = {}, {}
    for name, value in state.items():
        a = value.numpy()
        if scheme == 'float16':
            stored[name] = a.astype(np.float16)
            restored[name] = torch.from_numpy(stored[name])
        elif scheme.startswith('int8') and a.ndim == 2:
            axis = 1 if scheme == 'int8_per_row' else None
            scale = np.maximum(np.max(np.abs(a), axis=axis, keepdims=True) / 127, np.finfo(np.float32).tiny)
            q = np.clip(np.rint(a / scale), -127, 127).astype(np.int8)
            stored[name], stored[name + '_scale'] = q, scale
            restored[name] = torch.from_numpy(q.astype(np.float32) * scale)
        else:
            stored[name], restored[name] = a, value
    model = Policy().eval()
    if scheme == 'float16':
        model.half()
    model.load_state_dict(restored)
    raw = sum(a.nbytes for a in stored.values())
    return model, raw, artifact_size(stored)


def observations(game, model, count, seed):
    env, rng, chunks = GridGame(game, 64, seed), np.random.default_rng(seed + 1), []
    for step in range((count + 63) // 64):
        obs = env.observe().astype(np.float32)
        chunks.append(obs)
        # Include trained, scripted and random trajectories, not synthetic vectors.
        if step % 3 == 0:
            actions = model(torch.from_numpy(obs)).argmax(-1).numpy()
        elif step % 3 == 1:
            actions = scripted_actions(env, rng)
        else:
            actions = rng.integers(0, 5, 64)
        _, _, done = env.step(actions)
        env.reset(done)
    return torch.from_numpy(np.concatenate(chunks)[:count])


def latency(model, x, repeats, rounds):
    for _ in range(30):
        model(x)
    values = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            model(x)
        values.append((time.perf_counter_ns() - start) / repeats / 1000)
    return {'median_us_per_call': float(np.median(values)), 'min_us_per_call': min(values),
            'round_us_per_call': values, 'batch_size': len(x)}


class CastInput(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x.half()).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/policy_efficiency.json')
    parser.add_argument('--observations', type=int, default=8192)
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=500)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--seed', type=int, default=20261115)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    if 'qnnpack' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'qnnpack'
    report = {'config': vars(args), 'environment': {'python': platform.python_version(),
              'platform': platform.platform(), 'machine': platform.machine(),
              'torch': torch.__version__, 'numpy': np.__version__, 'threads': 1,
              'quantized_engines': torch.backends.quantized.supported_engines,
              'quantized_engine': torch.backends.quantized.engine},
              'notes': ['Int8 per-tensor/per-row variants quantize weights only; biases/scales float32; dequantization occurs once before timing.',
                        'dynamic_int8 uses native CPU int8 linear kernels and dynamically quantized activations. Accuracy is measured on the full corpus batch; activation quantization can depend on batch composition.',
                        'Float16 latency includes float32-input to float16 and output to float32 casts.',
                        'Artifact bytes use uncompressed NPZ for storage variants and torch.save(state_dict) for native int8; container overhead differs.',
                        'Greedy returns are separate rollouts with a shared seed; trajectory divergence prevents treating differences as paired causal estimates.',
                        'Latency measures model calls only, excludes game/environment/serialization and one-time loading; warm process, single thread; medians across rounds.'], 'games': {}}
    with torch.inference_mode():
        for game, path in CHECKPOINTS.items():
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            state = checkpoint['policy']
            baseline, _, _ = storage_variant(state, 'float32')
            x = observations(game, baseline, args.observations, args.seed)
            reference = baseline(x)
            result = {'checkpoint': str(path), 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'parameters': sum(v.numel() for v in state.values()),
                      'corpus_sha256': hashlib.sha256(x.numpy().tobytes()).hexdigest(), 'variants': {}}
            variants = []
            for scheme in ('float32', 'int8_per_tensor', 'int8_per_row', 'float16'):
                model, raw, size = storage_variant(state, scheme)
                if scheme == 'float16':
                    model = CastInput(model)
                variants.append((scheme, model, raw, size))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', DeprecationWarning)
                    native = torch.ao.quantization.quantize_dynamic(baseline, {nn.Linear}, dtype=torch.qint8)
                native(x[:64])
                stream = io.BytesIO()
                torch.save(native.state_dict(), stream)
                variants.append(('dynamic_int8', native, None, len(stream.getvalue())))
            except Exception as exc:
                result['native_int8_error'] = str(exc)
            for name, model, raw, size in variants:
                logits = model(x)
                error = (logits - reference).abs()
                result['variants'][name] = {'raw_tensor_bytes': raw, 'artifact_bytes': size,
                    'action_agreement': float((logits.argmax(-1) == reference.argmax(-1)).float().mean()),
                    'mean_absolute_logit_error': float(error.mean()), 'max_absolute_logit_error': float(error.max()),
                    'latency': {str(batch): latency(model, x[:batch], args.repeats, args.rounds) for batch in (1, 64)},
                    'greedy_evaluation': evaluate(game, model, episodes=args.episodes, seed=args.seed + 100, batch=64)}
            report['games'][game] = result
            print(game, json.dumps({k: {'agreement': v['action_agreement'], 'bytes': v['artifact_bytes'],
                  'latency_us_batch1': v['latency']['1']['median_us_per_call'],
                  'return': v['greedy_evaluation']['mean_return']} for k, v in result['variants'].items()}), flush=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(destination)


if __name__ == '__main__':
    main()
