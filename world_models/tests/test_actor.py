
import random

from world_models.actor import RandomActor

def test_random_actor():
    random.seed(0)
    actor = RandomActor([1, 2, 3])

    computed = [actor.get_action(None) for _ in range(5)]
    expected = [2, 2, 1, 2, 3]

    assert computed == expected
