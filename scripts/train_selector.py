"""Refit the 17-feature selector from saved train/validation NPZ states."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from orthrus_parallel.training import fit_classifier, predict, prompt_weights, softmax


def load_states(directory):
    states = {'train': [], 'validation': []}
    seen = set()
    hashes = {}
    for shard in sorted(directory.glob('shard*')):
        if json.loads((shard / 'receipt.json').read_text())['state'] != 'complete':
            raise ValueError(f'Incomplete collection: {shard}')
        for path in sorted(shard.glob('*.npz')):
            meta = json.loads(path.with_suffix('.json').read_text())
            if meta['id'] in seen or meta['split'] not in states:
                raise ValueError('Repeated prompt ID or unsupported split')
            seen.add(meta['id'])
            hashes[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as data:
                if data['x'].shape != (len(data['r']), 16, 17) or len(data['rank']) != len(data['r']):
                    raise ValueError('Expected aligned x [states,16,17], r and rank arrays')
                if not np.isfinite(data['x']).all():
                    raise ValueError('Nonfinite features')
                for features, boundary, rank in zip(data['x'], data['r'], data['rank']):
                    if not 0 <= boundary < 16 or not 0 <= rank <= 8:
                        raise ValueError('Label outside its supported range')
                    states[meta['split']].append({'x':features.astype(np.float64),
                        'r':int(boundary), 'rank':int(rank), 'task':meta['id']})
    if not all(states.values()):
        raise ValueError('Both train and validation data are required')
    return states, hashes


def examples(states, head):
    rows = []
    for state in states:
        if head == 'hazard':
            rows.extend((state['x'][j], int(j < state['r']), state['task'])
                        for j in range(min(state['r'] + 1, 15)))
        else:
            rows.append((state['x'][state['r']], state['rank'], state['task']))
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), [r[2] for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new directory')
    states, hashes = load_states(args.data)
    args.output.mkdir(parents=True)
    models, selection = {}, {}
    for head, classes in [('hazard', 2), ('rank', 9)]:
        train_x, train_y, train_ids = examples(states['train'], head)
        val_x, val_y, val_ids = examples(states['validation'], head)
        options = []
        for regularization in [.001, .01, .1]:
            model = fit_classifier(train_x, train_y, train_ids, classes, regularization)
            probabilities = predict(model, val_x)
            loss = float(prompt_weights(val_ids) @ -np.log(np.maximum(probabilities[np.arange(len(val_y)), val_y], 1e-12)))
            options.append((loss, regularization, model))
        loss, regularization, model = min(options, key=lambda value: value[0])
        np.savez(args.output / f'linear_{head}.npz', **model)
        models[head] = model
        selection[head] = {'regularization':regularization, 'validation_loss':loss,
            'train_examples':len(train_x), 'validation_examples':len(val_x)}
    posteriors = []
    for state in states['validation']:
        acceptance = predict(models['hazard'], state['x'][:-1])[:, 1]
        survival = np.r_[1, np.cumprod(acceptance)]
        boundary = np.r_[survival[:-1] * (1 - acceptance), survival[-1]]
        posteriors.append(boundary[:, None] * predict(models['rank'], state['x']))
    weights = prompt_weights([state['task'] for state in states['validation']])
    options = []
    for temperature in [.7, 1., 1.3, 1.7, 2.2]:
        calibrated = [softmax(np.log(np.maximum(p, 1e-12)).reshape(-1) / temperature).reshape(16, 9) for p in posteriors]
        loss = float(weights @ np.array([-np.log(p[state['r'], state['rank']])
            for p, state in zip(calibrated, states['validation'])]))
        options.append((loss, temperature))
    selection.update(temperature=min(options)[1], dimensions=17,
        scope='Trained from provided train/validation states; no evaluation prompts consumed')
    (args.output / 'selection.json').write_text(json.dumps(selection, indent=2))
    (args.output / 'training.json').write_text(json.dumps({'state':'complete','sources':hashes,
        'prompts':{split:len({s['task'] for s in values}) for split,values in states.items()},
        'states':{split:len(values) for split,values in states.items()},'selection':selection}, indent=2))
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()
