import numpy as np
import torch
import deep_util
from nonlinear_system import VanderPolEnv, rollout


class TestDeepLearning:
    def test_uplift(self):
        with torch.no_grad():
            state_dim = 3
            high_dim = 8
            hidden_size = 10
            time_len = 10
            network = deep_util.UpLiftNet(state_dim, hidden_size, high_dim)
            state_stack = torch.randn(time_len, state_dim, dtype=torch.float)
            highdim_state = network(state_stack)
            assert highdim_state.shape == (time_len, high_dim)

    def test_koopman(self):
        with torch.no_grad():
            state_dim, high_dim, hidden_size, action_dim = 3, 8, 10, 2
            time_len = 10
            network = deep_util.KoopmanStruct(state_dim, action_dim, hidden_size, high_dim)
            state_stack = torch.randn(time_len, state_dim, dtype=torch.float)
            action_stack = torch.randn(time_len, action_dim, dtype=torch.float)
            assert network.calc_z(state_stack).shape == (time_len, state_dim + high_dim)
            assert network.predict(state_stack, action_stack).shape == (time_len,state_dim + high_dim)

    def test_replay_buffer(self):
        state_dim, action_dim, capacity, sample_size = 2, 1, 100, 5
        buffer = deep_util.ReplayBuffer(state_dim, action_dim, capacity)
        env = VanderPolEnv()
        time_len = 45

        def random_policy(state):
            return 2 * np.random.random(1) - 1

        states, actions, next_states = rollout(env, random_policy, time_len,buffer=buffer)
        assert len(buffer) == time_len
        batch = buffer.sample(sample_size)
        assert batch['state'].shape == (sample_size, state_dim)
        assert batch['action'].shape == (sample_size, action_dim)
        assert batch['state_next'].shape == (sample_size, state_dim)



class TestNonlinear:
    def test_rollout(self):
        env = VanderPolEnv()

        def random_policy(state):
            return 2 * np.random.random(1) - 1

        time_len = 10
        states, actions, next_states = rollout(env, random_policy, time_len)
        assert states.shape == (time_len, env.state_dim)
        assert actions.shape == (time_len, env.action_dim)
        assert next_states.shape == (time_len, env.state_dim)
        assert np.allclose(states[1:], next_states[:-1])