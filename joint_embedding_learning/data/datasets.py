"""Paired tactile tensors, canonical sensor transforms, and episode splits."""
import glob
import os
import re
from collections import defaultdict
import torch
from torch.utils.data import Dataset, random_split, Subset
from torchvision import transforms
_EP_FR_RE = re.compile(r'ep(\d+)_fr(\d+)\.pt$')
_FLIPROT_NAMES = ('fliplr', 'flipud', 'rot180')

def _parse_ep_fr(path):
    """Parse 'ep0000_fr0032.pt' style filenames. Returns (episode_id, frame_id) or (-1, -1)."""
    m = _EP_FR_RE.search(os.path.basename(path))
    if m is None:
        return -1, -1
    return int(m.group(1)), int(m.group(2))

class CrossModalPairDataset(Dataset):
    """
    Single-pair dataset for tri-modal training (raw-tensor .pt format).

    Directory layout (per pair root):
        pair_root/sensor_a/ep****_fr****.pt
        pair_root/sensor_b/ep****_fr****.pt

    Each .pt file contains a *raw tensor* (NOT a dict):
        Daimon  : (3, 240, 320)  float32
        eFlesh  : (15,)          float32
        Flexi   : (12, 16)       float32

    Pairing rule: exact basename match (ep0000_fr0001.pt in sensor_a/ <-> same
    name in sensor_b/). Files that exist on only one side are dropped.
    """

    def __init__(self, pair_root, sensor_a, sensor_b,
                 transforms_a=None, transforms_b=None,
                 mean_a=None, std_a=None, mean_b=None, std_b=None,
                 device='cpu', objects=None, frame_stride=1):
        a_dir = os.path.join(pair_root, sensor_a)
        b_dir = os.path.join(pair_root, sensor_b)

        if objects is not None:
            # Per-object layout: sensor/{object}/ep****_fr****.pt
            a_files = {}
            b_files = {}
            for obj in objects:
                for p in glob.glob(os.path.join(a_dir, obj, '*.pt')):
                    a_files[f"{obj}/{os.path.basename(p)}"] = p
                for p in glob.glob(os.path.join(b_dir, obj, '*.pt')):
                    b_files[f"{obj}/{os.path.basename(p)}"] = p
        else:
            # Flat layout: sensor/ep****_fr****.pt
            a_files = {os.path.basename(p): p
                       for p in glob.glob(os.path.join(a_dir, '*.pt'))}
            b_files = {os.path.basename(p): p
                       for p in glob.glob(os.path.join(b_dir, '*.pt'))}
        common = sorted(set(a_files) & set(b_files), key=lambda name: (os.path.dirname(name), _parse_ep_fr(name)))

        self.sensor_a = sensor_a
        self.sensor_b = sensor_b
        self.files_a = [a_files[n] for n in common]
        self.files_b = [b_files[n] for n in common]
        self.episode_per_sample = []
        self.frame_per_sample = []
        # object id per sample: parsed from the "object/" key prefix used in
        # the per-object layout (flat layout -> all 0). Episode IDs are not
        # unique across objects, so evaluation retains both IDs.
        self.object_per_sample = []
        self.object_names = []
        _obj_to_id = {}
        for n in common:
            ep, fr = _parse_ep_fr(n)
            self.episode_per_sample.append(ep)
            self.frame_per_sample.append(fr)
            obj = n.split('/')[0] if '/' in n else '_flat_'
            if obj not in _obj_to_id:
                _obj_to_id[obj] = len(self.object_names)
                self.object_names.append(obj)
            self.object_per_sample.append(_obj_to_id[obj])

        # Per-episode temporal subsampling: keep every `frame_stride`-th frame
        # within each (object, episode). Used to down-sample a higher-rate
        # pair (e.g. ef = eFlesh+FlexiTac) so the three pairs have comparable
        # temporal density / dataset size.
        if frame_stride and frame_stride > 1:
            by_ep = defaultdict(list)
            for i in range(len(self.files_a)):
                by_ep[(self.object_per_sample[i],
                       self.episode_per_sample[i])].append(i)
            keep = []
            for idxs in by_ep.values():
                idxs.sort(key=lambda i: self.frame_per_sample[i])
                keep.extend(idxs[::frame_stride])
            keep.sort()
            self.files_a = [self.files_a[i] for i in keep]
            self.files_b = [self.files_b[i] for i in keep]
            self.episode_per_sample = [self.episode_per_sample[i] for i in keep]
            self.frame_per_sample = [self.frame_per_sample[i] for i in keep]
            self.object_per_sample = [self.object_per_sample[i] for i in keep]

        self.transforms_a = transforms_a
        self.transforms_b = transforms_b
        self.mean_a = torch.tensor(mean_a, dtype=torch.float32) if mean_a is not None else None
        self.std_a = torch.tensor(std_a, dtype=torch.float32) if std_a is not None else None
        self.mean_b = torch.tensor(mean_b, dtype=torch.float32) if mean_b is not None else None
        self.std_b = torch.tensor(std_b, dtype=torch.float32) if std_b is not None else None
        self.device = device

    def __len__(self):
        return len(self.files_a)

    @staticmethod
    def _apply_zscore(x, mean, std):
        if mean is None or std is None:
            return x
        m = mean.to(x.device)
        s = std.to(x.device)
        # broadcast over the leading dims; eFlesh: (15,) over (15,);
        # Daimon: (3,) over (3, H, W) requires reshape to (3, 1, 1).
        if x.dim() >= 2 and m.dim() == 1 and m.shape[0] == x.shape[0]:
            m = m.view(-1, *([1] * (x.dim() - 1)))
            s = s.view(-1, *([1] * (x.dim() - 1)))
        return (x - m) / (s + 1e-8)

    def _load_one(self, path, transforms_, mean, std):
        x = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(x, dict):
            # backward-compat: also accept dict-style files if encountered.
            if 'image' in x: x = x['image']
            elif 'magnetic' in x: x = x['magnetic']
            elif 'flexi' in x or 'flexitac' in x:
                x = x.get('flexi', x.get('flexitac'))
            else:
                raise ValueError(f"Unknown dict keys in {path}: {list(x.keys())}")
        x = x.float()
        x = self._apply_zscore(x, mean, std)
        if transforms_ is not None:
            x = transforms_(x)
        return x

    def __getitem__(self, idx):
        x_a = self._load_one(self.files_a[idx], self.transforms_a,
                             self.mean_a, self.std_a)
        x_b = self._load_one(self.files_b[idx], self.transforms_b,
                             self.mean_b, self.std_b)
        object_id = torch.tensor(self.object_per_sample[idx], dtype=torch.long)
        episode_id = torch.tensor(self.episode_per_sample[idx], dtype=torch.long)
        frame_id = torch.tensor(self.frame_per_sample[idx], dtype=torch.long)
        # label triple: [object_id, episode_id, frame_id]. Consumers that only
        # used [episode, frame] ignore labels entirely, so this is additive.
        return [x_a, x_b], [object_id, episode_id, frame_id]


class _RuntimeFlipRot:
    """Deterministic flip/rotation transform usable inside transforms.Compose.

        fliplr → mirror about the vertical axis  (left/right swap)
        flipud → mirror about the horizontal axis (top/bottom swap)
        rot180 → both, equivalent to undo-fliplr + apply-flipud

    Works for (C, H, W), (H, W), and (D,) tensors via negative-axis indices.
    """

    def __init__(self, mode):
        if mode not in _FLIPROT_NAMES:
            raise ValueError(f"Unknown flip/rot mode '{mode}', expected one of {_FLIPROT_NAMES}")
        self.mode = mode

    def __call__(self, x):
        if x.dim() < 2:
            return torch.flip(x, dims=[-1]) if self.mode == 'fliplr' else x
        if self.mode == 'fliplr':
            return torch.flip(x, dims=[-1])
        if self.mode == 'flipud':
            return torch.flip(x, dims=[-2])
        return torch.flip(x, dims=[-2, -1])

class _DeformOnlyGray:
    """Daimon channel reduction: keep channel 0 (normal deformation), drop the
    two shear channels, and replicate ch0 to 3 identical channels.

    The Daimon .pt stores (3, H, W) = [deform, shear_x, shear_y]. This makes
    the encoder see [deform, deform, deform] — a grayscale-deform input — so
    the shared latent represents only normal deformation (what FlexiTac, the
    normal-only sensor, can also represent).

    Applied AFTER z-score (CrossModalPairDataset z-scores the real 3-channel
    image first, so ch0 is correctly normalized with ch0's stats before it is
    selected here — no daimon_stats change needed).
    """

    def __call__(self, x):
        if x.dim() == 2:                 # (H, W) -> (1, H, W)
            x = x.unsqueeze(0)
        return x[0:1].repeat(3, 1, 1)    # (3, H, W), all = channel 0

def _build_sensor_transforms(dataset_details, sensor_name, image_size, extra_names=None):
    """Construct (transforms, mean, std) for a sensor, optionally appending
    deterministic flip/rot transforms named in `extra_names`."""
    transforms_list = []
    mean = None
    std = None

    if sensor_name == 'daimon':
        names = dataset_details.get('daimon_transforms', None) or []
        if 'resize' in names:
            transforms_list.append(transforms.Resize((image_size, image_size), antialias=True))
        if 'normalize' in names:
            stats = dataset_details.get('daimon_stats', None)
            if stats is not None:
                mean = stats['mean']
                std = stats['std']
        # 'deform_only': keep ch0 (deform), drop shear, replicate to 3 ch.
        # Runs after resize; z-score already applied per-channel in _load_one.
        if 'deform_only' in names:
            transforms_list.append(_DeformOnlyGray())
    elif sensor_name == 'eflesh':
        names = dataset_details.get('eflesh_transforms', None) or []
        if 'normalize' in names:
            stats = dataset_details.get('eflesh_stats', None)
            if stats is not None:
                mean = stats['mean']
                std = stats['std']
    elif sensor_name == 'flexi':
        names = dataset_details.get('flexitac_transforms', None) or []
        if 'normalize' in names:
            stats = dataset_details.get('flexitac_stats', None)
            if stats is not None:
                m = stats['mean']
                s = stats['std']
                mean = m if isinstance(m, (list, tuple)) else [float(m)]
                std = s if isinstance(s, (list, tuple)) else [float(s)]
    else:
        raise ValueError(f"Unknown sensor: {sensor_name}")

    # Flip/rot at the tail of the chain. z-score is applied separately in
    # CrossModalPairDataset._apply_zscore (per-pixel), and commutes with
    # these geometric ops.
    for nm in (extra_names or ()):
        if nm not in _FLIPROT_NAMES:
            raise ValueError(
                f"extra_transforms entry '{nm}' must be one of {_FLIPROT_NAMES}"
            )
        transforms_list.append(_RuntimeFlipRot(nm))

    transform = transforms.Compose(transforms_list) if transforms_list else None
    return transform, mean, std

def get_cross_modal_tri_datasets(dataset_name, dataset_details,
                                        data_split='train', device='cpu',
                                        objects=None, pair_strides=None):
    """Per-pair-transform-aware tri-modal dataset builder.

    Mirrors the shared get_cross_modal_tri_datasets but reads optional
    per-pair `extra_transforms_a` / `extra_transforms_b` from each pair's
    config.

    Returns:
        train_datasets : dict {'de': ds, 'ef': ds, 'fd': ds}
        val_datasets   : dict {'de': ds_val, 'ef': ds_val, 'fd': ds_val}
        empty_test     : {}
    """
    if dataset_details.get('dataset_type') != 'cross_modal_tri':
        raise ValueError('Expected dataset_type=cross_modal_tri')
    if set(dataset_details['pairs']) != {'de', 'ef', 'fd'}:
        raise ValueError('Dataset configuration must define de, ef, fd')
    data_root = dataset_details['data_root']
    image_size = dataset_details.get('image_size', 224)
    pairs = dataset_details['pairs']
    train_frac = float(dataset_details.get('train_frac', 0.8))

    train_datasets = {}
    val_datasets = {}

    print(f"[datasets] building '{dataset_name}' with per-pair transforms:")
    for pair_key, pair_cfg in pairs.items():
        sensor_a = pair_cfg['sensor_a']
        sensor_b = pair_cfg['sensor_b']
        pair_root = os.path.join(data_root, pair_cfg['root'])
        extra_a = pair_cfg.get('extra_transforms_a', None)
        extra_b = pair_cfg.get('extra_transforms_b', None)

        ta, ma, sa = _build_sensor_transforms(
            dataset_details, sensor_a, image_size, extra_names=extra_a,
        )
        tb, mb, sb = _build_sensor_transforms(
            dataset_details, sensor_b, image_size, extra_names=extra_b,
        )

        print(f"  [{pair_key}] {sensor_a:6s} ↔ {sensor_b:6s} | "
              f"extra_a={extra_a or '[]'}  extra_b={extra_b or '[]'}")

        stride = int((pair_strides or {}).get(pair_key, 1))
        if stride > 1:
            print(f"  [{pair_key}] frame_stride={stride} "
                  f"(keep every {stride}th frame per episode)")
        ds_full = CrossModalPairDataset(
            pair_root, sensor_a, sensor_b,
            transforms_a=ta, transforms_b=tb,
            mean_a=ma, std_a=sa, mean_b=mb, std_b=sb,
            device=device, objects=objects, frame_stride=stride,
        )

        # Train/val split. Two modes, selected by the dataset_details key
        # 'episode_val_split' (default False = original behavior):
        #
        #   False -> frame-level random_split (seed 0). Original behavior.
        #            Caveat: adjacent episode frames are near-duplicates, so
        #            train/val leak near-twins.
        #   True  -> episode-level split: whole (object, episode) groups go to
        #            train OR val, never split. Deterministic (sorted +
        #            modular): per object, every val_stride-th episode -> val,
        #            so every object is represented in val and there is no
        #            near-duplicate leakage.
        #            If 'val_episode_ids' is set (list of int), those specific
        #            episode indices go to val regardless of stride logic.
        if not bool(dataset_details.get('episode_val_split', False)):
            n = len(ds_full)
            train_len = int(train_frac * n)
            val_len = n - train_len
            ds_train, ds_val = random_split(
                ds_full, [train_len, val_len],
                generator=torch.Generator().manual_seed(0),
            )
            train_datasets[pair_key] = ds_train
            val_datasets[pair_key] = ds_val
        else:
            groups = defaultdict(list)        # (obj_id, ep_id) -> sample idxs
            for i in range(len(ds_full)):
                groups[(ds_full.object_per_sample[i],
                        ds_full.episode_per_sample[i])].append(i)
            by_obj = defaultdict(list)        # obj_id -> [(obj,ep) keys]
            for key in groups:
                by_obj[key[0]].append(key)
            train_idx, val_idx = [], []

            explicit_val_eps = dataset_details.get('val_episode_ids', None)
            if explicit_val_eps is not None:
                # Two accepted shapes:
                #   list           — same val episode ids applied to every object
                #   dict {obj: []} — per-object val ids (needed when objects
                #                    have different episode counts, e.g. the
                #                    10-object 'dataset_new' entry)
                if isinstance(explicit_val_eps, dict):
                    per_obj_val_sets = {}
                    for i, name in enumerate(ds_full.object_names):
                        ids = explicit_val_eps.get(name, [])
                        per_obj_val_sets[i] = set(int(e) for e in ids)
                    missing = [n for n in ds_full.object_names
                               if n not in explicit_val_eps]
                    if missing:
                        print(f"  [{pair_key}] WARNING val_episode_ids has no "
                              f"entry for objects {missing} -> all their "
                              f"episodes will go to train")
                    for key, idxs in groups.items():
                        obj_id, ep_id = key
                        if ep_id in per_obj_val_sets.get(obj_id, set()):
                            val_idx.extend(idxs)
                        else:
                            train_idx.extend(idxs)
                    val_eps_per_obj = {
                        ds_full.object_names[i]: sorted(per_obj_val_sets[i])
                        for i in per_obj_val_sets
                    }
                    print(f"  [{pair_key}] episode-level split (explicit, "
                          f"per-object): {len(train_idx)} train / "
                          f"{len(val_idx)} val frames over {len(groups)} "
                          f"episodes (val eps = {val_eps_per_obj})")
                else:
                    val_ep_set = set(int(e) for e in explicit_val_eps)
                    for key, idxs in groups.items():
                        if key[1] in val_ep_set:
                            val_idx.extend(idxs)
                        else:
                            train_idx.extend(idxs)
                    print(f"  [{pair_key}] episode-level split (explicit): "
                          f"{len(train_idx)} train / {len(val_idx)} val frames "
                          f"over {len(groups)} episodes "
                          f"(val eps = {sorted(val_ep_set)})")
            else:
                val_stride = max(2, round(1.0 / max(1e-6, 1.0 - train_frac)))
                for obj_id in sorted(by_obj):
                    for j, key in enumerate(sorted(by_obj[obj_id])):
                        (val_idx if j % val_stride == 0 else train_idx).extend(
                            groups[key])
                print(f"  [{pair_key}] episode-level split: "
                      f"{len(train_idx)} train / {len(val_idx)} val frames "
                      f"over {len(groups)} episodes (val = every {val_stride}th)")

            train_idx.sort()
            val_idx.sort()
            train_datasets[pair_key] = Subset(ds_full, train_idx)
            val_datasets[pair_key] = Subset(ds_full, val_idx)

    return train_datasets, val_datasets, {}
