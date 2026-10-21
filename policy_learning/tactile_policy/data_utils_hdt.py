import random
import numpy as np
import torch
import json
import torchvision
import os
import h5py
from torch.utils.data import DataLoader
import cv2
import torchvision.transforms.v2

# from tactile_policy.pointcloud_utils import PointcloudAugmentation
from tactile_policy.dataset_sampler import get_episode_sampling_prob

def interpolate_128dim_action(action_nd, target_m):
    """
    Interpolate action from $n$ actions to $m$ actions.
    Works with any action dimension (dynamically uses input shape).
    """
    target_t = np.linspace(0, action_nd.shape[0] - 1, target_m)

    interpolated_action = np.zeros((target_m, action_nd.shape[1]))
    for idx in range(target_m):
        curr_t = target_t[idx]
        src_prev_t = int(curr_t)
        src_next_t = min(src_prev_t + 1, action_nd.shape[0] - 1)
        assert src_prev_t >= 0 and src_next_t < action_nd.shape[0], "Index out of bounds: {} {} {}".format(idx, src_prev_t, src_next_t)
        src_prev_action = action_nd[src_prev_t]
        src_next_action = action_nd[src_next_t]
        # This interpolates 3D positions / qpos
        interpolated_action[idx] = (src_next_action - src_prev_action) * (curr_t - src_prev_t) + src_prev_action
        # NOTE(roger): conceptually, 6D rotation is not a linear space, so we need to use SLERP
        # however, we tried simple interpolation on the robot and it worked well
    return interpolated_action

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_config,
                 hdf_path_list,
                 camera_names,
                 chunk_size,
                 episode_len_list,
                 task_episode_cnt,
                 visual_preprocessor,
                 train=True,
                 tactile_names=None,
                 tactile_modality='none', num_workers=4):
        super(EpisodicDataset).__init__()
        self.camera_names = camera_names
        self.tactile_names = tactile_names if tactile_names else []
        self.tactile_modality = tactile_modality
        self.train = train
        self.data_config = data_config
        self.episode_len_list = episode_len_list
        self.dataset_paths = hdf_path_list
        self.task_episode_cnt = task_episode_cnt
        self.visual_preprocessor = visual_preprocessor
        self.chunk_size = chunk_size  # action length (e.g., chunk size)

        self.predict_delta_action = data_config["predict_delta_action"]
        self.load_hdf_to_cpu = data_config["load_hdf_to_cpu"]
        self.cond_mask_prob = data_config["cond_mask_prob"]
        self.slow_down_factor = data_config["human_slow_down_factor"]
        self.num_samples_for_delta_action = data_config['num_samples_for_delta_action']


        self.cumulative_len = np.cumsum(self.episode_len_list)

        self.sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in self.episode_len_list])


        if self.load_hdf_to_cpu:
            self.cached_hdf_dict = {}
            for single_hdf_path in self.dataset_paths:
                if single_hdf_path not in self.cached_hdf_dict:
                    self.cached_hdf_dict[single_hdf_path] = {}
                    with h5py.File(single_hdf_path, 'r') as root:
                        self.cached_hdf_dict[single_hdf_path]['observation.state'] = root['observation.state'][()]
                        self.cached_hdf_dict[single_hdf_path]['observation.tac0'] = root['observation.tac0'][()]
                        self.cached_hdf_dict[single_hdf_path]['observation.tac1'] = root['observation.tac1'][()]
                        for cam_name in self.camera_names:
                            images = root[f'observation.image.{cam_name}'][()]
                            if len(images.shape) == 4 and images.shape[-1] == 3:
                                images = images.transpose(0, 3, 1, 2)
                            self.cached_hdf_dict[single_hdf_path][f'observation.image.{cam_name}'] = images
                        for tactile_name in self.tactile_names:
                            if f'observation.image.{tactile_name}' in root:
                                tactile_images = root[f'observation.image.{tactile_name}'][()]
                                if len(tactile_images.shape) == 4 and tactile_images.shape[-1] == 3:
                                    tactile_images = tactile_images.transpose(0, 3, 1, 2)
                                self.cached_hdf_dict[single_hdf_path][f'observation.image.{tactile_name}'] = tactile_images
                        self.cached_hdf_dict[single_hdf_path]["action"] =  root["action"][()]
                        self.cached_hdf_dict[single_hdf_path]['attrs'] = {k: v for k, v in root.attrs.items()}
        
        self.training_transforms = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.RandomCrop(size = (int(self.data_config["image_resolution_hw"][0] * 0.95), int(self.data_config["image_resolution_hw"][1] * 0.95))),
            torchvision.transforms.v2.Resize(size = (self.data_config["image_resolution_hw"][0], self.data_config["image_resolution_hw"][1])),
            torchvision.transforms.v2.RandomRotation(degrees = (-5, 5)),
            torchvision.transforms.v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            # torchvision.transforms.v2.GaussianBlur(kernel_size=(9,9), sigma=(0.1,2.0)),
        ])

        self.norm_stats, self.embodiment_list = self.get_norm_stats()

        self.episode_sampling_prob = get_episode_sampling_prob(self.task_episode_cnt,
                                                               self.embodiment_list,
                                                               self.episode_len_list,
                                                               data_config["sampler_type"])

        if self.predict_delta_action:
            delta_action_norm_stats = self.get_delta_action_norm_stats()
            for emb in self.norm_stats:
                self.norm_stats[emb]['action_mean'] = delta_action_norm_stats[emb]['delta_action_mean']
                self.norm_stats[emb]['action_std'] = delta_action_norm_stats[emb]['delta_action_std']
    
    def get_norm_stats(self):
        if hasattr(self, 'norm_stats'):
            return self.norm_stats
        
        norm_stats_dict = {}
        embodiment_list = []
        # First, gather types of embodiments
        for single_hdf_path in self.dataset_paths:
            with h5py.File(single_hdf_path, 'r') as root:
                emb_key = root.attrs.get('embodiment', 'default')
                norm_stats_dict[emb_key] = {
                    "actions": [],
                    "states": [],
                    "pointclouds": [],
                    "tactile_arrays": []
                }
                embodiment_list.append(emb_key)
        print(f"Found embodiments: {norm_stats_dict.keys()}")

        for hdf_path in self.dataset_paths:
            with h5py.File(hdf_path, 'r') as root:
                emb_key = root.attrs.get('embodiment', 'default')
                state = root['observation.state'][()]
                action = root['action'][()]
                norm_stats_dict[emb_key]["actions"].append(torch.from_numpy(action))
                norm_stats_dict[emb_key]["states"].append(torch.from_numpy(state))
                # Collect tactile arrays only when modality is 'array'
                if self.tactile_modality == 'array':
                    for tactile_name in self.tactile_names:
                        if f'observation.{tactile_name}' in root:
                            tactile_array = root[f'observation.{tactile_name}'][()]
                            norm_stats_dict[emb_key]["tactile_arrays"].append(torch.from_numpy(tactile_array))
               

        SAME_NORMALIZATION = True
        if SAME_NORMALIZATION:
            all_actions = []
            all_states = []
            all_pointclouds = []
            all_tactile_arrays = []
            for emb in norm_stats_dict:
                all_actions.append(torch.cat(norm_stats_dict[emb]['actions']))
                all_states.append(torch.cat(norm_stats_dict[emb]['states']))
                if norm_stats_dict[emb]['tactile_arrays']:
                    all_tactile_arrays.append(torch.cat(norm_stats_dict[emb]['tactile_arrays']))


            all_actions = torch.cat(all_actions)
            all_states = torch.cat(all_states)
            if all_tactile_arrays:
                all_tactile_arrays = torch.cat(all_tactile_arrays)


            # normalize action data
            action_mean = all_actions.mean(dim=0, keepdim=True).numpy().squeeze()
            action_std = all_actions.std(dim=0, keepdim=True)
            action_std = torch.clip(action_std, 1e-2, np.inf).numpy().squeeze()
            # if self.predict_delta_action:
            #     action_mean = np.zeros_like(action_mean)

            # normalize qpos data
            qpos_mean = all_states.mean(dim=0, keepdim=True).numpy().squeeze()
            qpos_std = all_states.std(dim=0, keepdim=True)
            qpos_std = torch.clip(qpos_std, 1e-2, np.inf).numpy().squeeze()

            # normalize tactile array data
            if all_tactile_arrays is not None and len(all_tactile_arrays) > 0:
                tactile_mean = all_tactile_arrays.mean(dim=0, keepdim=True).numpy().squeeze()
                tactile_std = all_tactile_arrays.std(dim=0, keepdim=True)
                tactile_std = torch.clip(tactile_std, 1e-2, np.inf).numpy().squeeze()
            else:
                tactile_mean = None
                tactile_std = None

            for emb in norm_stats_dict:
                norm_stats_dict[emb]['action_mean'] = action_mean
                norm_stats_dict[emb]['action_std'] = action_std

                norm_stats_dict[emb]['qpos_mean'] = qpos_mean
                norm_stats_dict[emb]['qpos_std'] = qpos_std

                norm_stats_dict[emb]['tactile_mean'] = tactile_mean
                norm_stats_dict[emb]['tactile_std'] = tactile_std

                del norm_stats_dict[emb]['actions']
                del norm_stats_dict[emb]['states']
                del norm_stats_dict[emb]['pointclouds']
                del norm_stats_dict[emb]['tactile_arrays']
        else:
            for emb in norm_stats_dict:
                norm_stats_dict[emb]['actions'] = torch.cat(norm_stats_dict[emb]['actions'])
                norm_stats_dict[emb]['states'] = torch.cat(norm_stats_dict[emb]['states'])

                # normalize action data
                norm_stats_dict[emb]['action_mean'] = norm_stats_dict[emb]['actions'].mean(dim=0, keepdim=True).numpy().squeeze()
                norm_stats_dict[emb]['action_std'] = norm_stats_dict[emb]['actions'].std(dim=0, keepdim=True)
                norm_stats_dict[emb]['action_std'] = torch.clip(norm_stats_dict[emb]['action_std'], 1e-2, np.inf).numpy().squeeze()
                # if self.predict_delta_action:
                #     norm_stats_dict[emb]['action_mean'] = np.zeros_like(norm_stats_dict[emb]['action_mean'])

                # normalize qpos data
                norm_stats_dict[emb]['qpos_mean'] = norm_stats_dict[emb]['states'].mean(dim=0, keepdim=True).numpy().squeeze()
                norm_stats_dict[emb]['qpos_std'] = norm_stats_dict[emb]['states'].std(dim=0, keepdim=True)
                norm_stats_dict[emb]['qpos_std'] = torch.clip(norm_stats_dict[emb]['qpos_std'], 1e-2, np.inf).numpy().squeeze()

                # normalize tactile array data
                if norm_stats_dict[emb]['tactile_arrays']:
                    norm_stats_dict[emb]['tactile_arrays'] = torch.cat(norm_stats_dict[emb]['tactile_arrays'])
                    norm_stats_dict[emb]['tactile_mean'] = norm_stats_dict[emb]['tactile_arrays'].mean(dim=0, keepdim=True).numpy().squeeze()
                    norm_stats_dict[emb]['tactile_std'] = norm_stats_dict[emb]['tactile_arrays'].std(dim=0, keepdim=True)
                    norm_stats_dict[emb]['tactile_std'] = torch.clip(norm_stats_dict[emb]['tactile_std'], 1e-2, np.inf).numpy().squeeze()
                else:
                    norm_stats_dict[emb]['tactile_mean'] = None
                    norm_stats_dict[emb]['tactile_std'] = None

                del norm_stats_dict[emb]['actions']
                del norm_stats_dict[emb]['states']
                del norm_stats_dict[emb]['tactile_arrays']

        return norm_stats_dict, embodiment_list
    
    def get_delta_action_norm_stats(self):
        delta_action_norm_stats_dict = {}
        for emb in self.norm_stats:
            delta_action_norm_stats_dict[emb] = {
                'delta_action': []
            }
        for i in range(self.num_samples_for_delta_action):
            image_data, qpos_data, action_data, is_pad, conditioning_list, _, _ = self.__getitem__(i)
            embodiment = conditioning_list['embodiment']
            uaction = action_data * (self.norm_stats[embodiment]["action_std"]+ 1e-6) + self.norm_stats[embodiment]["action_mean"]
            uqpos = qpos_data * (self.norm_stats[embodiment]["qpos_std"]+ 1e-6) +  + self.norm_stats[embodiment]["qpos_mean"]
            delta_action = uaction - uqpos
            delta_action_norm_stats_dict[embodiment]['delta_action'].append(delta_action)
        
        for emb in self.norm_stats:
            delta_action_norm_stats_dict[emb]['delta_action'] = torch.cat(delta_action_norm_stats_dict[emb]['delta_action'], dim = 0)
            delta_action_norm_stats_dict[emb]['delta_action_mean'] = delta_action_norm_stats_dict[emb]['delta_action'].mean(dim=0, keepdim=True).numpy().squeeze()
            delta_action_norm_stats_dict[emb]['delta_action_std'] = delta_action_norm_stats_dict[emb]['delta_action'].std(dim=0, keepdim=True)
            delta_action_norm_stats_dict[emb]['delta_action_std'] = torch.clip(delta_action_norm_stats_dict[emb]['delta_action_std'], 1e-2, np.inf).numpy().squeeze()
            del delta_action_norm_stats_dict[emb]['delta_action']

        return delta_action_norm_stats_dict
    
    def __len__(self):
        return 100000000

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len_list[episode_index])
        return episode_index, start_ts

    def read_one(self, index, start_ts):
        episode_len = self.episode_len_list[index]

        start_ts = np.random.choice(episode_len)

        single_hdf_path = self.dataset_paths[index]

        # ===== Load HDF file =====
        if self.load_hdf_to_cpu:
            root = self.cached_hdf_dict[single_hdf_path]
        else:
            root = h5py.File(single_hdf_path, 'r')

        qpos = root['observation.state'][start_ts]

        image_dict = dict()
        for cam_name in self.camera_names:
            if isinstance(root[f'observation.image.{cam_name}'], bytes):
                actual_cam_name = root[f'observation.image.{cam_name}'].decode('utf-8')
                image_dict[cam_name] = root[f'{actual_cam_name}'][start_ts]
            elif root[f'observation.image.{cam_name}'].dtype == 'O':
                # A trick we use to avoid explicitly replicating images.
                # TODO(roger): do it in a better way.
                actual_cam_name = root[f'observation.image.{cam_name}'][()].decode('utf-8')
                image_dict[cam_name] = root[f'{actual_cam_name}'][start_ts]
            else:
                image_dict[cam_name] = root[f'observation.image.{cam_name}'][start_ts]

            if len(image_dict[cam_name].shape) == 1:
                # Compressed JPEG format images are represented as (N,) uint8 array. N is different for every image.
                image_dict[cam_name] = cv2.imdecode(image_dict[cam_name], cv2.IMREAD_COLOR)
                assert image_dict[cam_name].shape == (self.data_config["image_resolution_hw"][0], self.data_config["image_resolution_hw"][1], 3)
                # Images are in RGB (verified by plt.imshow) and HWC in this case
                image_dict[cam_name] = image_dict[cam_name].transpose(2, 0, 1)  # HWC -> CHW
            elif len(image_dict[cam_name].shape) == 3:
                # Check if image needs transpose from HWC to CHW
                # If already in CHW format (from cache), shape is (C, H, W) where C=3
                # If in HWC format (from file), shape is (H, W, C) where C=3
                if image_dict[cam_name].shape[2] == 3:
                    # Image is in HWC format - need to transpose to CHW for PyTorch
                    image_dict[cam_name] = image_dict[cam_name].transpose(2, 0, 1)  # HWC -> CHW

        all_time_action = root["action"][start_ts:start_ts+self.chunk_size]

        # Load tactile images (for image mode) BEFORE closing file
        all_tactile_images = []
        if self.tactile_names:
            for tactile_name in self.tactile_names:
                if f'observation.image.{tactile_name}' in root:
                    tactile_img = root[f'observation.image.{tactile_name}'][start_ts]
                    if len(tactile_img.shape) == 1:
                        tactile_img = cv2.imdecode(tactile_img, cv2.IMREAD_COLOR)
                        tactile_img = tactile_img.transpose(2, 0, 1)
                    elif len(tactile_img.shape) == 3 and tactile_img.shape[2] == 3:
                        tactile_img = tactile_img.transpose(2, 0, 1)
                    all_tactile_images.append(tactile_img)

        # Load tactile arrays only when modality is 'array'
        all_tactile_arrays = []
        if self.tactile_modality == 'array' and self.tactile_names:
            for tactile_name in self.tactile_names:
                if f'observation.{tactile_name}' in root:
                    tactile_array = root[f'observation.{tactile_name}'][start_ts]
                    tactile_array = np.atleast_1d(tactile_array)
                    all_tactile_arrays.append(tactile_array)

        cur_pointcloud = None

        if not self.load_hdf_to_cpu:
            lang_instruction = root.attrs.get('description', '')
            embodiment = root.attrs.get('embodiment', 'default')
            root.close()
        else:
            lang_instruction = root['attrs']['description'] if 'attrs' in root and 'description' in root['attrs'] else ''
            embodiment = root['attrs']['embodiment'] if 'attrs' in root and 'embodiment' in root['attrs'] else 'default'

        if "human" in embodiment:
            SLOW_DOWN_FACTOR = self.slow_down_factor
            all_time_action = interpolate_128dim_action(all_time_action, all_time_action.shape[0] * SLOW_DOWN_FACTOR)
            all_time_action = all_time_action[:self.chunk_size]

        padded_action = np.zeros((self.chunk_size, all_time_action.shape[1]), dtype=np.float32)
        padded_action[:all_time_action.shape[0], :] = all_time_action

        real_len = episode_len - start_ts

        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[real_len:] = True

        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        if self.tactile_modality == 'image' and self.tactile_names and not all_tactile_images:
            raise ValueError(f"Tactile modality is 'image' but no tactile images found in {single_hdf_path}")
        if self.tactile_modality in ['array', 'state'] and self.tactile_names and not all_tactile_arrays:
            raise ValueError(f"Tactile modality is '{self.tactile_modality}' but no tactile arrays found in {single_hdf_path}")

        image_data = self.visual_preprocessor(all_cam_images)
        if self.train:
            image_data = self.training_transforms(image_data)

        # Process tactile images (for image mode)
        if all_tactile_images:
            all_tactile_images = np.stack(all_tactile_images, axis=0)
            tactile_images = self.visual_preprocessor(all_tactile_images)
            if self.train:
                tactile_images = self.training_transforms(tactile_images)
        else:
            tactile_images = torch.tensor([])

        # Process tactile arrays (for array/state modes)
        if all_tactile_arrays:
            all_tactile_arrays = np.stack(all_tactile_arrays, axis=0)
            tactile_arrays = torch.from_numpy(all_tactile_arrays).float()
            # Apply Gaussian normalization to tactile arrays (no conditioning dropout)
            if self.norm_stats[embodiment]["tactile_mean"] is not None:
                tactile_arrays = (tactile_arrays - self.norm_stats[embodiment]["tactile_mean"]) / (self.norm_stats[embodiment]["tactile_std"] + 1e-6)
                tactile_arrays = torch.clamp(tactile_arrays, -5.0, 5.0)
        else:
            tactile_arrays = torch.tensor([])

        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        if self.predict_delta_action:
            action_data = action_data - qpos_data

        action_data = (action_data - self.norm_stats[embodiment]["action_mean"]) / (self.norm_stats[embodiment]["action_std"] + 1e-6)
        qpos_data = (qpos_data - self.norm_stats[embodiment]["qpos_mean"]) / (self.norm_stats[embodiment]["qpos_std"] + 1e-6) \
            if random.random() > self.cond_mask_prob else torch.zeros_like(qpos_data)

        if random.random() < self.cond_mask_prob:
            lang_instruction = ''


        conditioning_dict = {
            "plain_text": lang_instruction,
            'embodiment': embodiment,
        }

        conditioning_dict["pointcloud"] = torch.tensor([])

        return image_data, qpos_data, action_data, is_pad, conditioning_dict, tactile_images, tactile_arrays
    
    def __getitem__(self, _idx):
        episode_idx = np.random.choice(len(self.episode_len_list), p=self.episode_sampling_prob)
        ts_index = np.random.randint(self.sum_dataset_len_l[episode_idx], self.sum_dataset_len_l[episode_idx + 1])

        # index: index of the selected episode ID, start_ts: start timestep within the episode
        index, start_ts = self._locate_transition(ts_index)
        return self.read_one(index, start_ts)

def gather_hdf_paths(base_dir, task_names):
    all_hdf_paths = []
    task_episode_cnt = []
    for task_name in task_names:
        task_dir = os.path.join(base_dir, task_name)
        print(f"Task dir: {task_dir}")
        assert os.path.exists(task_dir) and os.path.isdir(task_dir)
        cur_task_cnt = 0
        for fn in sorted(os.listdir(task_dir)):
            if fn.endswith('.hdf5'):
                all_hdf_paths.append(os.path.join(task_dir, fn))
                cur_task_cnt += 1
        task_episode_cnt.append(cur_task_cnt)
    return all_hdf_paths, task_episode_cnt

def get_all_episode_len(hdf_list):
    all_episode_len = []
    for hdf_path in hdf_list:
        with h5py.File(hdf_path, 'r') as root:
            qpos = root['observation.state'][()]
        all_episode_len.append(len(qpos))

    return all_episode_len

def collate_fn(batch):
    image_data, qpos_data, action_data, is_pad, conditioning_list, tactile_images_list, tactile_arrays_list = zip(*batch)
    image_data = torch.stack(image_data)
    qpos_data = torch.stack(qpos_data)
    action_data = torch.stack(action_data)
    is_pad = torch.stack(is_pad)

    # Stack tactile images if any exist
    if any(t.numel() > 0 for t in tactile_images_list):
        tactile_images = torch.stack([t if t.numel() > 0 else torch.zeros_like(tactile_images_list[0]) for t in tactile_images_list])
    else:
        tactile_images = torch.tensor([])

    # Stack tactile arrays if any exist
    if any(t.numel() > 0 for t in tactile_arrays_list):
        tactile_arrays = torch.stack([t if t.numel() > 0 else torch.zeros_like(tactile_arrays_list[0]) for t in tactile_arrays_list])
    else:
        tactile_arrays = torch.tensor([])

    KEYWORDS_LIST = ['plain_text', 'pointcloud', 'embodiment']
    ret_conditioning_dict = {}
    for keyword in KEYWORDS_LIST:
        cur_conditioning_list = [conditioning[keyword] for conditioning in conditioning_list]
        if keyword == 'language_embeddings':
            if any(cond is None for cond in cur_conditioning_list):
                continue
            cur_conditioning_len = [arr.shape[0] for arr in cur_conditioning_list]
            cur_conditioning_tensor = torch.nn.utils.rnn.pad_sequence(
                    cur_conditioning_list,
                    batch_first=True,
                    padding_value=0)
            valid_mask = torch.zeros(
                cur_conditioning_tensor.shape[0], cur_conditioning_tensor.shape[1], dtype=torch.bool)
            for i, l in enumerate(cur_conditioning_len):
                valid_mask[i, :l] = True

            ret_conditioning_dict[keyword] = cur_conditioning_tensor
            ret_conditioning_dict[keyword + '_mask'] = valid_mask
        elif keyword == 'plain_text' or keyword == 'embodiment':
            ret_conditioning_dict[keyword] = cur_conditioning_list
        elif keyword == 'pointcloud':
            ret_conditioning_dict[keyword] = torch.stack(cur_conditioning_list, dim=0)

    return image_data, qpos_data, action_data, is_pad, ret_conditioning_dict, tactile_images, tactile_arrays

def load_data(base_dir,
              data_config,
              dataset_json_path: str,
              camera_names,
              chunk_size,
              batch_size_train,
              batch_size_val,
              visual_preprocessor,
              jax_dataset=False,
              tactile_names=None,
              tactile_modality='none', num_workers=4):
    
    
    assert os.path.exists(dataset_json_path)

    with open(dataset_json_path, 'r') as f:
        dataset_config = json.load(f)
    
    dataset_dict = {}
    
    for split in dataset_config:
        assert split in ['train', 'val'], "Only train and val splits now supported, you gave {}".format(split)
        is_train = (split == 'train')
        task_names = dataset_config[split]
        task_names = sorted(task_names)
        hdf_path_list, task_episode_cnt = gather_hdf_paths(base_dir, task_names)
        assert hdf_path_list == sorted(hdf_path_list)

        print("Total {} episodes for {} split".format(len(hdf_path_list), split))
        if not hdf_path_list:
            raise ValueError(f'No episodes found for {split}: {base_dir}/{task_names}')
        all_episode_len = get_all_episode_len(hdf_path_list)
        if min(all_episode_len) < 2:
            raise ValueError('Each episode needs at least two frames for normalization')
        # jax transforms training data in the forward function
        if jax_dataset:
            train = False
        else:
            train = is_train

        dataset_dict[split] = EpisodicDataset(data_config,
                                    hdf_path_list,
                                    camera_names,
                                    chunk_size,
                                    all_episode_len,
                                    task_episode_cnt,
                                    visual_preprocessor,
                                    train=train,
                                    tactile_names=tactile_names,
                                    tactile_modality=tactile_modality)
    
    train_dataset = dataset_dict['train']
    val_dataset = dataset_dict['val']

    val_dataset.norm_stats = train_dataset.get_norm_stats()
    if jax_dataset:
        train_dataloader = train_dataset
        val_dataloader = val_dataset
    else:   
        train_dataloader = DataLoader(train_dataset, pin_memory=True, num_workers=num_workers, batch_size=batch_size_train, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, pin_memory=True, num_workers=num_workers, batch_size=batch_size_val, collate_fn=collate_fn)

    norm_stats = train_dataset.get_norm_stats()

    return train_dataloader, val_dataloader, norm_stats

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        try:
            new_d[k] = v.detach()
        except AttributeError:
            new_d[k] = v
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
