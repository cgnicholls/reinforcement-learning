
import numpy as np

from world_models.environment import Pong


def test_pong_preprocess():
    env = Pong()

    initial_state = env.reset()

    assert initial_state.shape == (64, 64, 3)

    state, _, done, _ = env.step(1)

    assert state.shape == (64, 64, 3)

    _, _, done, _ = env.step(1)
    _, _, done, _ = env.step(1)
    _, _, done, _ = env.step(1)
    state2, _, done, _ = env.step(1)

    assert state2.shape == (64, 64, 3)

    assert np.any(state != state2)

    assert np.min(state) >= 0.0
    assert np.max(state) <= 1.0

#
# def test_pong_render():
#     env = Pong()
#
#     env.render()
#
#     initial_state = env.reset()
#
#     assert initial_state.shape == (64, 64, 3)
#
#     state, _, done, _ = env.step(1)
#
#     assert state.shape == (64, 64, 3)
#
#     while not done:
#         state, _, done, _ = env.step(np.random.choice([0, 1]))
#         env.render()
