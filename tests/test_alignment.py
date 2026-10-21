import copy
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torchvision.transforms.functional import resize

from conftest import ROOT, run_cli
from joint_embedding_learning.data.datasets import get_cross_modal_tri_datasets
from joint_embedding_learning.training import _tri_zip_loaders, _infonce_alpha
from joint_embedding_learning.utils import models_setup
from joint_embedding_learning.utils.utils import get_loss, load_optimizer


def test_pairing_split_and_orientation(paired_data):
    _, root, _, details = paired_data
    train, val, _ = get_cross_modal_tri_datasets('dataset_new', details, objects=['circle'])
    for pair in ('de', 'ef', 'fd'):
        assert len(train[pair]) == 4 and len(val[pair]) == 3
        assert {train[pair][i][1][1].item() for i in range(4)} == {0}
        assert {val[pair][i][1][1].item() for i in range(3)} == {1}
    raw = torch.load(root / 'de_pair/daimon/circle/ep0000_fr0000.pt', weights_only=True)
    stats = details['daimon_stats']
    expected = (raw - torch.tensor(stats['mean'])[:, None, None]) / (torch.tensor(stats['std'])[:, None, None] + 1e-8)
    expected = torch.flip(resize(expected, [224, 224], antialias=True), [-2, -1])
    torch.testing.assert_close(train['de'][0][0][0], expected)
    assert train['de'][0][0][0].device.type == 'cpu'


def test_loader_restarts_without_retaining_batches():
    class CountingLoader:
        def __init__(self, length): self.length, self.starts = length, 0
        def __len__(self): return self.length
        def __iter__(self):
            self.starts += 1
            yield from range(self.length)
    loaders = {p: CountingLoader(n) for p, n in [('de', 3), ('ef', 1), ('fd', 2)]}
    assert len(list(_tri_zip_loaders(loaders))) == 3
    assert loaders['ef'].starts == 3
    with pytest.raises(ValueError, match='Empty'):
        list(_tri_zip_loaders(dict(loaders, ef=CountingLoader(0))))


@pytest.mark.parametrize('epoch,start,warmup,expected', [(0,0,0,1), (0,9999,0,0), (2,2,0,1), (3,2,2,.5)])
def test_alignment_schedule(epoch, start, warmup, expected):
    args = SimpleNamespace(current_epoch=epoch, infonce_warmup_start=start, infonce_warmup_epochs=warmup)
    assert _infonce_alpha(args) == expected


def test_optimizer_honors_weight_decay():
    opt, _, _ = load_optimizer(SimpleNamespace(optimizer='Adam', learning_rate=.001, weight_decay=.012), torch.nn.Linear(2,2))
    assert opt.param_groups[0]['weight_decay'] == .012


@pytest.mark.parametrize('name', ['tactx', 'reconstruction', 'contrastive', 'l2'])
def test_documented_training_checkpoints(checkpoints, name):
    path = checkpoints[name]
    model_config = yaml.safe_load((path / 'model_details.yaml').read_text())
    dataset_config = yaml.safe_load((path / 'dataset_details.yaml').read_text())
    assert dataset_config['objects'] == ['circle']
    assert dataset_config['val_episode_ids'] == {'circle': [1]}
    model = models_setup.get_model(model_config['model_name'], model_config)
    state = torch.load(path / 'model.pt', weights_only=True)
    model.load_state_dict(state, strict=True)
    assert all(torch.isfinite(t).all() for t in state.values())
    torch.manual_seed(model_config['seed'])
    initial = models_setup.get_model(model_config['model_name'], model_config).state_dict()
    assert any(not torch.equal(state[k], initial[k]) for k in state if state[k].is_floating_point())


@pytest.mark.parametrize('name,arch', [('cross_modal_vae3','residual'), ('cross_modal_vae3','small_cnn'), ('cross_modal_clr3','small_cnn')])
def test_original_checkpoint_architecture_compatibility(monkeypatch, name, arch):
    # Historical classes are read only, not copied back into the release.
    source = subprocess.check_output(['git','show','f5d66f82:joint_embedding_learning/cross_modal/model.py'], cwd=ROOT, text=True)
    namespace = {}
    exec(compile(source, '<original model>', 'exec'), namespace)
    config = yaml.safe_load((ROOT / 'config/models_config.yaml').read_text())[name]
    config['flexitac_arch'] = arch
    torch.manual_seed(8)
    model = models_setup.get_model(name, config).eval()
    cls = 'CrossModalVAE3' if name == 'cross_modal_vae3' else 'CrossModalCLR3'
    monkeypatch.setattr(models_setup, cls, namespace[cls])
    old = models_setup.get_model(name, config).eval()
    old.load_state_dict(model.state_dict(), strict=True)
    x, y = torch.randn(2,15), torch.randn(2,12,16)
    with torch.no_grad():
        torch.manual_seed(11); current = model.forward_pair(x,y,'ef')
        torch.manual_seed(11); previous = old.forward_pair(x,y,'ef')
    for key in current:
        if torch.is_tensor(current[key]): torch.testing.assert_close(current[key], previous[key], rtol=0, atol=0)


@pytest.mark.parametrize('variant', ['tactx', 'contrastive'])
def test_evaluation_cli(checkpoints, paired_data, variant):
    work, processed, _, _ = paired_data
    output = work / f'eval_{variant}'
    extra = ['--contrastive_only'] if variant == 'contrastive' else []
    text = run_cli('scripts.evaluate', '--checkpoint_dir', checkpoints[variant], '--device', 'cpu',
                   '--out_dir', output, '--batch_size', 2, '--max_tsne_points', 24, *extra)
    assert (output / 'metrics.txt').is_file(), text
    assert list(output.glob('*.png'))
    assert 'nan' not in (output / 'metrics.txt').read_text().lower()


def test_contact_and_dynamic_evaluation(checkpoints, paired_data):
    work, processed, _, _ = paired_data
    labels = work / 'contact_labels'
    run_cli('scripts.build_contact_labels', '--data_root', processed, '--out_root', labels, '--objects', 'circle')
    assert len(list(labels.rglob('*.pt'))) == 6
    run_cli('scripts.evaluate', '--checkpoint_dir', checkpoints['tactx'], '--device', 'cpu',
            '--out_dir', work / 'eval_optional', '--batch_size', 2, '--max_tsne_points', 24,
            '--contact_label_root', labels, '--dynamic_data_root', processed, '--dynamic_objects', 'circle')


def test_stats_find_object_subdirectories(paired_data):
    _, processed, _, _ = paired_data
    output = run_cli('scripts.compute_tri_stats', '--data_root', processed)
    assert 'daimon_stats:' in output and 'eflesh_stats:' in output and 'flexitac_stats:' in output


def test_worker_loading_matches_single_process(paired_data):
    from torch.utils.data import DataLoader
    _, _, _, details = paired_data
    train, _, _ = get_cross_modal_tri_datasets('dataset_new', details, objects=['circle'])
    one = next(iter(DataLoader(train['de'], batch_size=2, num_workers=0)))
    workers = next(iter(DataLoader(train['de'], batch_size=2, num_workers=2)))
    for a,b in zip(one[0], workers[0]): torch.testing.assert_close(a,b)
    for a,b in zip(one[1], workers[1]): torch.testing.assert_close(a,b)


def test_invalid_or_too_small_training_data_fails(paired_data):
    from joint_embedding_learning.training import train_tri
    _, _, _, details = paired_data
    train, val, _ = get_cross_modal_tri_datasets('dataset_new', details, objects=['circle'])
    config = yaml.safe_load((ROOT/'config/models_config.yaml').read_text())['cross_modal_vae3']
    with pytest.raises(ValueError, match='need at least'):
        train_tri(train, {'val':val}, 'dataset_new', details, 'cross_modal_vae3', config, None, device='cpu')
