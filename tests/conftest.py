"""Small real-format fixtures; no project datasets or published weights needed."""
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, WANDB_MODE='disabled', MPLBACKEND='Agg', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
torch.set_num_threads(2)


def run_cli(module, *args, cwd=ROOT):
    result = subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=cwd,
                            env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=300)
    assert result.returncode == 0, result.stdout[-16000:]
    return result.stdout


@pytest.fixture(scope='session')
def paired_data(tmp_path_factory):
    work = tmp_path_factory.mktemp('paired')
    raw, processed = work / 'raw', work / 'processed'
    rng = np.random.default_rng(42)
    sensor_shapes = {'daimon': (5, 240, 320), 'eflesh': (5, 3), 'flexi': (12, 16)}
    for left, right in [('daimon', 'eflesh'), ('eflesh', 'flexi'), ('flexi', 'daimon')]:
        for ep, frames in [(0, 4), (1, 3)]:
            folder = raw / f'left_{left}_right_{right}_circle' / f'episode_{ep:04d}'
            folder.mkdir(parents=True)
            for frame in range(frames):
                payload = {key: rng.normal(size=sensor_shapes[sensor]).astype('float32') * (frame + 1) * 0.02
                           for key, sensor in [('s0', left), ('s1', right)]}
                with (folder / f'{frame:06d}.pkl').open('wb') as f:
                    pickle.dump(payload, f)
    run_cli('scripts.preprocess', '--input_root', raw, '--output_root', processed)
    details = yaml.safe_load((ROOT / 'config/datasets_config.yaml').read_text())['dataset_new']
    details.update(data_root=str(processed), val_episode_ids={'circle': [1]}, objects=['circle'])
    config = work / 'datasets.yaml'
    config.write_text(yaml.safe_dump({'dataset_new': details}))
    return work, processed, config, details


@pytest.fixture(scope='session')
def checkpoints(paired_data):
    work, processed, config, details = paired_data
    result = {}
    variants = {
        'tactx': [],
        'reconstruction': ['--infonce_warmup_start', '9999'],
        'contrastive': ['--model_name', 'cross_modal_clr3'],
        'l2': ['--alignment_loss', 'l2'],
    }
    for name, extra in variants.items():
        output = run_cli('scripts.train_alignment', '--dataset_config', config,
                        '--dataset_name', 'dataset_new', '--data_root', processed,
                        '--device', 'cpu', '--objects', 'circle', '--epochs', 1,
                        '--batch_size', 2, '--num_workers', 0, '--run_name', name,
                        '--checkpoint_root', work / 'checkpoints', *extra)
        path = Path(output.split('Checkpoint directory: ')[-1].strip())
        assert (path / 'model.pt').is_file(), output
        result[name] = path
    return result
