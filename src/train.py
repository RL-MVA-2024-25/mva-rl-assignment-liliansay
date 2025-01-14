import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import gc
from huggingface_hub import hf_hub_download, HfApi
import joblib

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.env = env
        self.gamma = 0.9
        self.Qfunctions = []

    def collect_samples(self, horizon, disable_tqdm=False, print_done_states=False, exploration_prob=0.2):
        s, _ = self.env.reset()
        S, A, R, S2, D = [], [], [], [], []

        for _ in tqdm(range(horizon), disable=disable_tqdm):
            if np.random.rand() < exploration_prob or not self.Qfunctions:
                a = self.act(s, use_random=True)
            else:
                a = self.act(s, use_random=False)
            
            s2, r, done, trunc, _ = self.env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)

            if done or trunc:
                s, _ = self.env.reset()
                if done and print_done_states:
                    print("Episode done!")
            else:
                s = s2

        S = np.array(S)
        A = np.array(A).reshape((-1, 1))
        R = np.array(R)
        S2 = np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D

    def train(self, nb_iter, initial_horizon=10000, incremental_horizon=1000, exploration_prob=0.2):
        S, A, R, S2, D = self.collect_samples(horizon=initial_horizon)
        SA = np.append(S, A, axis=1)

        for iter in tqdm(range(nb_iter), desc="FQI Iterations"):
            if iter == 0:
                value = R.copy()
            else:
                Q2 = np.zeros((S.shape[0], self.env.action_space.n))
                for a2 in range(self.env.action_space.n):
                    A2 = a2 * np.ones((S2.shape[0], 1))
                    S2A2 = np.append(S2, A2, axis=1)
                    Q2[:, a2] = self.Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = R + self.gamma * (1 - D) * max_Q2

            Q = RandomForestRegressor()
            Q.fit(SA, value)
            self.Qfunctions = [Q]

            new_S, new_A, new_R, new_S2, new_D = self.collect_samples(
                horizon=incremental_horizon, disable_tqdm=True, exploration_prob=exploration_prob
            )
            S = np.vstack((S, new_S))
            A = np.vstack((A, new_A))
            R = np.append(R, new_R)
            S2 = np.vstack((S2, new_S2))
            D = np.append(D, new_D)
            SA = np.append(S, A, axis=1)

            gc.collect()

    def act(self, observation, use_random=False):
        if use_random:
            return self.env.action_space.sample()
        else:
            Qsa = []
            for a in range(self.env.action_space.n):
                sa = np.append(observation,a).reshape(1, -1)
                Qsa.append(self.Qfunctions[-1].predict(sa))
            return np.argmax(Qsa)

    def save(self, path):
        joblib.dump(self.Qfunctions[-1], f"./model/{path}")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=f"./model/{path}",
            path_in_repo=f"{path}",
            repo_id="liliansay/random-forest-model",
            repo_type="model",
        )

    def load(self):
        REPO_ID = "liliansay/random-forest-model"
        FILENAME = "trained_agent.joblib"

        model = joblib.load(
            hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        )
        self.Qfunctions.append(model)
