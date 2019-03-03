
import numpy as np

from world_models.environment import Pong


def test_pong_preprocess():
    env = Pong()

    initial_state = env.reset()

    assert initial_state.shape == (64, 64, 3)

    state, _, done, _ = env.step(2)

    assert state.shape == (64, 64, 3)

    state2, _, done, _ = env.step(2)

    assert state2.shape == (64, 64, 3)

    assert np.any(state != state2)

    assert np.min(state) >= 0.0
    assert np.max(state) <= 1.0
