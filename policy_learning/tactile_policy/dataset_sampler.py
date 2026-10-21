import numpy as np
from collections import defaultdict

DEFINED_DISTRIBUTION = {
    'actionnet_gr1t1_6dof': 1,
    'human_egodex': 1,
    'human_avp': 2,
    'human_zed': 2,
    'h1_inspire': 2,
    'human_aria': 2,
}

def get_episode_sampling_prob(task_episode_cnt_list, embodiment_list, episode_len_list, sampler_type):
    if sampler_type == 'uniform':
        return None  # when it is None, the default behavior is uniform sampling
    elif sampler_type == 'norm_by_defined_distribution':
        task_idx_group = defaultdict(list)
        cur_idx = 0
        for task_idx in range(len(task_episode_cnt_list)):
            task_episode_cnt = task_episode_cnt_list[task_idx]
            cur_embodiment = embodiment_list[cur_idx]
            assert embodiment_list[cur_idx:cur_idx + task_episode_cnt] == [cur_embodiment] * task_episode_cnt
            task_idx_group[cur_embodiment].extend(list(range(cur_idx, cur_idx + task_episode_cnt)))
            cur_idx += task_episode_cnt
        assert set(task_idx_group.keys()).issubset(DEFINED_DISTRIBUTION.keys())
        prob_arr = np.ones(len(episode_len_list)) / len(episode_len_list)
        for emb, idx_list in task_idx_group.items():
            prob_arr[idx_list] = DEFINED_DISTRIBUTION[emb] / len(idx_list)
        assert not np.isclose(np.sum(prob_arr), 0)

        prob_arr = prob_arr / np.sum(prob_arr)
        return prob_arr
    # =======================================
    # NOTE(Roger): deprecated. Use only as potential baselines
    # =======================================
    elif sampler_type == 'norm_by_embodiment':
        P_ROBOT = 0.5
        P_HUMAN = 1 - P_ROBOT
        human_idx_list = []
        robot_idx_list = []
        cur_idx = 0
        for task_idx in range(len(task_episode_cnt_list)):
            task_episode_cnt = task_episode_cnt_list[task_idx]
            cur_embodiment = embodiment_list[cur_idx]
            assert embodiment_list[cur_idx:cur_idx + task_episode_cnt] == [cur_embodiment] * task_episode_cnt
            if "human" in cur_embodiment:
                human_idx_list.extend(list(range(cur_idx, cur_idx + task_episode_cnt)))
            else:
                robot_idx_list.extend(list(range(cur_idx, cur_idx + task_episode_cnt)))
            cur_idx += task_episode_cnt
        assert len(embodiment_list) == len(episode_len_list)
        prob_arr = np.ones(len(episode_len_list)) / len(episode_len_list)
        if len(human_idx_list) > 0:
            prob_arr[human_idx_list] = P_HUMAN / len(human_idx_list)
        prob_arr[robot_idx_list] = P_ROBOT / len(robot_idx_list)
        if not np.isclose(np.sum(prob_arr), 1):
            print("=========")
            print(f"Warning: sum of prob_arr is not 1: {np.sum(prob_arr)}. Is only one embodiment available?")
            prob_arr = prob_arr / np.sum(prob_arr)
        return prob_arr
    elif sampler_type == 'norm_by_embodiment_and_task':
        P_ROBOT = 0.5
        P_HUMAN = 1 - P_ROBOT
        human_task_idx_group = []
        robot_task_idx_group = []
        cur_idx = 0
        for task_idx in range(len(task_episode_cnt_list)):
            task_episode_cnt = task_episode_cnt_list[task_idx]
            cur_embodiment = embodiment_list[cur_idx]
            assert embodiment_list[cur_idx:cur_idx + task_episode_cnt] == [cur_embodiment] * task_episode_cnt
            if "human" in cur_embodiment:
                human_task_idx_group.append(list(range(cur_idx, cur_idx + task_episode_cnt)))
            else:
                robot_task_idx_group.append(list(range(cur_idx, cur_idx + task_episode_cnt)))
            cur_idx += task_episode_cnt
        assert len(embodiment_list) == len(episode_len_list)
        prob_arr = np.ones(len(episode_len_list)) / len(episode_len_list)
        for task_idx_list in human_task_idx_group:
            prob_arr[task_idx_list] = P_HUMAN / (len(task_idx_list) * len(human_task_idx_group))
        for task_idx_list in robot_task_idx_group:
            prob_arr[task_idx_list] = P_ROBOT / (len(task_idx_list) * len(robot_task_idx_group))
        if not np.isclose(np.sum(prob_arr), 1):
            print("=========")
            print(f"Warning: sum of prob_arr is not 1: {np.sum(prob_arr)}. Is only one embodiment available?")
            prob_arr = prob_arr / np.sum(prob_arr)
        return prob_arr
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
