from pathlib import Path
import json
import pickle

import h5py
import numpy as np
import pytest
import torch
import yaml

from conftest import ROOT, run_cli
from joint_embedding_learning.utils.models_setup import get_model
from scripts.convert_tactile_to_latent import process_episode, preprocess_eflesh, preprocess_flexi, encode_batch


@pytest.fixture(scope='session')
def latent_model(checkpoints):
    path = checkpoints['tactx']
    config = yaml.safe_load((path / 'model_details.yaml').read_text())
    model = get_model('cross_modal_vae3', config).eval()
    model.load_state_dict(torch.load(path / 'model.pt', weights_only=True), strict=True)
    details = yaml.safe_load((path / 'dataset_details.yaml').read_text())
    return model, details


def make_episode(path, sensor='F', seed=5):
    rng = np.random.default_rng(seed)
    width = {'D': 384000, 'E': 15, 'F': 192}[sensor]
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.attrs['embodiment'] = 'robot'
        f.attrs['description'] = 'synthetic integration fixture'
        f.attrs['tactile_dim'] = width
        f.create_dataset('observation.state', data=rng.normal(size=(4,128)).astype('float32'))
        f.create_dataset('action', data=rng.normal(size=(4,128)).astype('float32'))
        for cam in ['cam0','cam1']:
            f.create_dataset(f'observation.image.{cam}', data=rng.integers(0,256,(4,240,320,3),dtype='uint8'), compression='gzip')
        for tac in ['tac0','tac1']:
            f.create_dataset(f'observation.{tac}', data=rng.normal(size=(4,width)).astype('float32') * .02).attrs['stream'] = tac
        f.create_dataset('scalar_metadata', data=7)
        f.create_group('metadata').attrs['retained'] = True
        f['metadata'].create_dataset('timestamps', data=np.arange(4))


@pytest.mark.parametrize('sensor', ['D','E','F'])
def test_conversion_cli_and_hdf_preservation(tmp_path, checkpoints, latent_model, sensor):
    src = tmp_path/'source/episode_0000.hdf5'
    make_episode(src, sensor)
    dest = tmp_path/'output/episode_0000.hdf5'
    run_cli('scripts.convert_tactile_to_latent', '--checkpoint_dir', checkpoints['tactx'],
            '--sensor', sensor, '--input_dir', src.parent, '--output_dir', dest.parent,
            '--device', 'cpu', '--batch_size', 3)
    with h5py.File(src) as before, h5py.File(dest) as after:
        assert after['observation.tac0'].shape == (4,16)
        assert after['observation.tac1'].shape == (4,16)
        assert np.isfinite(after['observation.tac0'][()]).all()
        np.testing.assert_array_equal(before['action'][()], after['action'][()])
        assert after['observation.image.cam0'].compression == 'gzip'
        assert after['scalar_metadata'][()] == 7
        assert after['metadata'].attrs['retained']
        assert after.attrs['embodiment'] == 'robot'
        assert after.attrs['tactile_dim'] == 16
        assert after['observation.tac1'].attrs['stream'] == 'tac1'
        if sensor == 'F':
            model, details = latent_model
            expected = encode_batch(model, preprocess_flexi(before['observation.tac0'][()], details), 'F', 'cpu')
            np.testing.assert_allclose(after['observation.tac0'][()], expected, rtol=2e-5, atol=2e-6)


def test_eflesh_transform_never_reorders_frames():
    x = np.arange(60, dtype='float32').reshape(4,15)
    for mode in ['none','flipud','rot180']:
        torch.testing.assert_close(preprocess_eflesh(x, {}, mode), torch.from_numpy(x))
    torch.testing.assert_close(preprocess_eflesh(x, {}, 'fliplr'), torch.from_numpy(x).flip(-1))


def test_conversion_rejects_overwrites_and_preserves_input(tmp_path, latent_model):
    model, details = latent_model
    src = tmp_path/'episode_0000.hdf5'
    make_episode(src)
    contents = src.read_bytes()
    with pytest.raises(ValueError, match='must differ'):
        process_episode(src, src, model, 'F', details, 224, 2, 'cpu')
    assert src.read_bytes() == contents
    dest = tmp_path/'out.hdf5'
    dest.write_bytes(b'existing output')
    with pytest.raises(FileExistsError):
        process_episode(src, dest, model, 'F', details, 224, 2, 'cpu')
    with h5py.File(src, 'a') as f:
        del f['observation.tac1']
    with pytest.raises(ValueError, match='missing'):
        process_episode(src, dest, model, 'F', details, 224, 2, 'cpu', overwrite=True)
    assert dest.read_bytes() == b'existing output'


def test_act_training_and_reload(tmp_path, latent_model):
    from tactile_policy.modeling.modeling_act import ACTPolicy
    from tactile_policy.data_utils_hdt import load_data
    from tactile_policy.modeling.utils import unit_normalization
    model, details = latent_model
    for split in ['train','val']:
        raw = tmp_path/f'raw_{split}/episode_0000.hdf5'
        make_episode(raw, seed=5 if split == 'train' else 7)
        process_episode(raw, tmp_path/f'data/{split}/episode_0000.hdf5', model,'F',details,224,3,'cpu')
    split_json = tmp_path/'split.json'
    split_json.write_text(json.dumps({'train':['train'], 'val':['val']}))
    common = ['--batch_size',1,'--num_epochs',2,'--lr',1e-5,'--chunk_size',4,
              '--exptid','smoke','--dataset_json',split_json,'--model_cfg','act_cross_modal_pnp_z16',
              '--data_root',tmp_path/'data','--checkpoint_root',tmp_path/'checkpoints',
              '--no_wandb','--cpu','--no_pretrained_backbone','--num_workers',0,
              '--val_batches',1,'--save_every',1]
    run_cli('tactile_policy.main', *common, cwd=tmp_path)
    checkpoint = tmp_path/'checkpoints/smoke_ckpt'
    assert (checkpoint/'policy_last.ckpt').is_file()
    config = json.loads((checkpoint/'policy_config.json').read_text())
    policy = ACTPolicy(config).eval()
    policy.load_state_dict(torch.load(checkpoint/'policy_last.ckpt',weights_only=True),strict=True)
    data_cfg = yaml.safe_load((ROOT/'policy_learning/tactile_policy/configs/data_base.yaml').read_text())['data']
    _, val, stats = load_data(str(tmp_path/'data'), data_cfg, str(split_json), ['cam0','cam1'],4,1,1,
                             unit_normalization,tactile_names=['tac0','tac1'],tactile_modality='array',num_workers=0)
    with (checkpoint/'dataset_stats.pkl').open('rb') as f:
        saved_stats=pickle.load(f)
    np.testing.assert_array_equal(saved_stats['robot']['tactile_mean'],stats['robot']['tactile_mean'])
    image,qpos,_,_,conditioning,_,tactile=next(iter(val))
    with torch.no_grad():
        actions=policy(image,qpos,conditioning_dict=conditioning,tactile_arrays=tactile)
    assert actions.shape == (1,4,128) and torch.isfinite(actions).all()
    # Resume actual optimizer/scheduler state from a periodic checkpoint.
    del policy
    run_cli('tactile_policy.main', *common, '--resume_from',checkpoint/'policy_iter_1_seed_0',cwd=tmp_path)


def test_daimon_channel_order_and_normalization():
    from scripts.convert_tactile_to_latent import preprocess_daimon
    raw = np.broadcast_to(np.array([10,20,30,40,50],dtype='float32'),(2,240,320,5)).copy().reshape(2,-1)
    result = preprocess_daimon(raw, {'daimon_stats': {'mean':[1,2,3], 'std':[1,2,4]}}, 224)
    assert result.shape == (2,3,224,224)
    for channel, expected in enumerate([9,19,11.75]):
        torch.testing.assert_close(result[:,channel], torch.full((2,224,224), expected, dtype=torch.float32))
