import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from nonlinear_system import NonLinearEnv, rollout


class UpLiftNet(nn.Module):
    def __init__(self, input_size, hidden_size, highdim_size):
        super(UpLiftNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, highdim_size)
        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leak_relu(self.fc1(x))
        x = self.leak_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class KoopmanStruct:
    def __init__(self, state_dim, action_dim, hidden_size, high_dim):
        self.input_dim = state_dim
        self.hidden_dim = hidden_size
        self.control_dim = action_dim
        self.high_dim = high_dim
        self.uplift = UpLiftNet(self.input_dim, self.hidden_dim, self.high_dim)
        self.K = torch.ones([self.input_dim + self.high_dim, self.input_dim + self.high_dim])
        self.L = torch.ones([self.input_dim + self.high_dim, self.control_dim])

    def calc_z(self, x):
        z = torch.cat((x, self.uplift(x)), 1)  # timelen*12
        return z

    def predict(self, x, u):
        z = self.calc_z(x)
        z_next = self.K.matmul(z.T) + self.L.matmul(u.T)  # timelen*12转化为12*timelen
        return z_next.T


class ReplayBuffer:
    def __init__(self, state_dim=2, action_dim=1, capacity=10000):
        self.state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.ptr, self.size, self.capacity = 0, 0, capacity

    def push(self, state, action, state_next):
        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.next_state_buf[self.ptr] = state_next
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.choice(self.size, batch_size, replace=False)
        batch = dict(state=self.state_buf[ind],
                     action=self.action_buf[ind],
                     state_next=self.next_state_buf[ind])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def __len__(self):
        return self.size


class TrainThread:
    def __init__(self, env: NonLinearEnv, buffer: ReplayBuffer):
        self.env = env
        hidden_dim, self.high_dim = 10, 10
        self.koopman = KoopmanStruct(env.state_dim, env.action_dim, hidden_dim, self.high_dim)
        self.optimizer = optim.RMSprop(self.koopman.uplift.parameters(), lr=0.001)
        self.criterion = nn.SmoothL1Loss()
        self.buffer = buffer

    def l2_regularization(self, l2_alpha):
        l2_loss = []
        for module in self.koopman.uplift.modules():
            if type(module) is nn.Linear:
                l2_loss.append((module.weight ** 2).sum())
        return l2_alpha * sum(l2_loss)

    def solve_koopman_kl(self):
        all_data = self.buffer.sample(self.buffer.size)
        all_x = all_data['state']
        all_u = all_data['action']
        all_next_x = all_data['state_next']
        with torch.no_grad():
            lift_all_x = self.koopman.calc_z(all_x)
            lift_all_next_x = self.koopman.calc_z(all_next_x)
            KL = lift_all_next_x.T.matmul(torch.linalg.pinv(torch.cat([lift_all_x.T, all_u.T])))
            self.koopman.K = KL[:, 0:self.env.state_dim + self.high_dim]
            self.koopman.L = KL[:, self.env.state_dim + self.high_dim:]

    def optimize_uplift_network(self, batch):
        """
        :param batch: 一个batch，batch_size*total_dim，包含x,u,next_x
        """
        state = batch['state']
        u = batch['action']
        next_state = batch['state_next']
        next_z_hat = self.koopman.predict(state, u)
        next_z = self.koopman.calc_z(next_state)
        self.optimizer.zero_grad()
        loss = self.criterion(next_z_hat, next_z) + \
               self.criterion(next_z_hat[:, 0:self.env.state_dim], next_z[:, 0:self.env.state_dim])
        loss /= state.shape[0]
        loss += self.l2_regularization(1e-6)
        loss.backward()
        self.optimizer.step()

    def test_error(self, render=False):
        test_time_len = 100

        def random_policy(state):
            return 2 * np.random.random(1) - 1

        self.env.reset(2 * np.random.random(2) - 1)
        states, actions, next_states = rollout(self.env, random_policy, test_time_len,
                                               buffer=self.buffer)  # TODO 如果test数据进入buffer，过拟合程度会变大

        with torch.no_grad():
            state_pred = torch.zeros(states.shape)
            state_pred[0] = torch.from_numpy(states[0])
            actions = torch.from_numpy(actions)
            for i in range(test_time_len - 1):
                state = state_pred[i]
                u = actions[i]
                pred = self.koopman.predict(state.unsqueeze(0), u.unsqueeze(0))
                state_pred[i + 1] = pred[0, 0:self.env.state_dim]

        if render:
            plt.plot(states[:, 0])
            plt.plot(state_pred[:, 0])
            plt.legend(['real', 'pred'])
            plt.show()
        return np.mean(np.abs(state_pred.numpy() - states))

    def run(self, batch_size=100, epoch=2000):
        error = []
        for i in range(epoch):
            batch = self.buffer.sample(batch_size)
            self.optimize_uplift_network(batch)
            if i % 1 == 0:
                self.solve_koopman_kl()

            if i % 50 == 0:
                total_error = 0
                eval_num = 5
                for _ in range(eval_num):
                    total_error += self.test_error(render=False)
                error.append(total_error / eval_num)

            if i % (epoch // 10) == 0:
                self.test_error(render=True)

        plt.plot(error)
        plt.ylim(0, 2)
        plt.title('error-epoch')
        plt.show()
