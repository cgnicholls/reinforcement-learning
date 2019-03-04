
from world_models.main import get_last_rollout_index


def test_get_last_rollout_index():
    rollout_file_names = ['rollouts_{}.h5'.format(i) for i in [1, 2, 7, 11, 77, 60, 101, 5]]
    assert get_last_rollout_index(rollout_file_names) == 101

    rollout_file_names = ['rollouts_{}.h5'.format(i) for i in [1, 2, 7]]
    assert get_last_rollout_index(rollout_file_names) == 7
