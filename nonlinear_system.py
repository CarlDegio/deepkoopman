from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt


class NonLinearEnv:
    def __init__(self, state_dim, action_dim, state0, dt):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = state0
        self.dt = dt
        self.t = 0

    @abstractmethod
    def step(self, u):
        pass

    def reset(self, state0=np.array(2)):  # TODO: default value
        self.state = state0
        self.t = 0


def rollout(env: NonLinearEnv, policy, time_len=100, buffer=None):
    state0 = 2 * np.random.random(2) - 1
    env.reset(state0)
    states = np.zeros((time_len, env.state_dim),dtype=np.float32)
    actions = np.zeros((time_len, env.action_dim),dtype=np.float32)
    next_states = np.zeros((time_len, env.state_dim),dtype=np.float32)
    for t in range(time_len):
        states[t]=env.state
        action = policy(env.state)
        actions[t]=action
        env.step(action)
        next_states[t]=env.state
        if buffer is not None:
            buffer.push(states[t], actions[t], next_states[t])
    return states, actions, next_states


class VanderPolEnv(NonLinearEnv):
    def __init__(self, state_dim=2, action_dim=1, state0=np.zeros(2), mu=0.5, dt=0.1):
        super().__init__(state_dim, action_dim, state0, dt)
        self.mu = mu

    def step(self, u):
        x = self.state[0]
        xd = self.state[1]
        xdd = self.mu * (1 - x ** 2) * xd - x + u

        xd_next = xd + xdd * self.dt
        x_next = x + xd_next * self.dt
        self.state = np.concatenate([x_next, xd_next])
        self.t = self.t + self.dt
        return self.state


class MassDampSpringEnv(NonLinearEnv):
    def __init__(self, state_dim=2, action_dim=1, state0=np.zeros(2), m=1, b=1, k0=1, k1=1, dt=0.1):
        super().__init__(state_dim, action_dim, state0, dt)
        self.m = m
        self.b = b
        self.k0 = k0
        self.k1 = k1
        self.dt = dt

    def step(self, u):
        x = self.state[0]
        xd = self.state[1]
        xdd = (-self.k1 * xd ** 3 - self.k0 * x - self.b * xd * abs(xd) + u) / self.m

        xd_next = xd + xdd * self.dt
        x_next = x + xd_next * self.dt
        self.state = np.concatenate([x_next, xd_next])
        self.t = self.t + self.dt
        return self.state


class RewardEnv(NonLinearEnv):
    def __init__(self, state_dim, action_dim, state0=np.zeros(2), dt=0.1):
        super().__init__(state_dim, action_dim, state0, dt)

    def step(self, u):
        xd_next = 1 if u > 0 else -1
        x_next = 1 if u > 0 else 0
        self.state = np.array([x_next, xd_next])
        self.t = self.t + self.dt
        return self.state


if __name__ == "__main__":
    vanderpol = VanderPolEnv()
    X = []
    T = []
    for i in range(100):
        state = vanderpol.step(-1)
        X.append(state)
        T.append(vanderpol.t)
    X = np.array(X)
    T = np.array(T)
    plt.plot(T, X[:, 0])
    plt.show()
