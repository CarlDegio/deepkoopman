import numpy as np
import deep_util
from nonlinear_system import VanderPolEnv, MassDampSpringEnv, RewardEnv, rollout

env = VanderPolEnv(mu=1)
# env=RewardEnv()

def collect_data(replay_buffer: deep_util.ReplayBuffer):
    def random_policy(state):
        return 2 * np.random.random(1) - 1

    for traj_num in range(20):
        env.reset(2 * np.random.random(2) - 1)
        rollout(env, random_policy, 100, buffer=replay_buffer)


if __name__ == "__main__":
    buffer = deep_util.ReplayBuffer(capacity=5000)
    collect_data(buffer)
    trainer = deep_util.TrainThread(env,buffer)
    trainer.solve_koopman_kl()
    trainer.run()
    trainer.test_error(render=True)
