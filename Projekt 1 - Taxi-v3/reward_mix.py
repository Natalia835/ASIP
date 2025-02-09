import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_function):
        super().__init__(env)
        self.reward_function = reward_function

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        # Modyfikacja nagrody
        reward = self.reward_function(state, reward, done, action)
        return state, reward, done, truncated, info

# Polityka epsilon-greedy
def choose_action(state, q_table, epsilon, action_space_size):
    if np.random.rand() < epsilon:
        return np.random.randint(action_space_size)  
    else:
        return np.argmax(q_table[state])            

def train_agent(env, alpha, gamma, epsilon, epsilon_decay, epsilon_min, episodes, max_steps):
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n

    q_table = np.zeros((state_space_size, action_space_size))
    rewards = []
    epsilon_values = []
    steps_per_episode = []

    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action = choose_action(state, q_table, epsilon, action_space_size)
            next_state, reward, done, truncated, info = env.step(action)

            # Aktualizacja Q-Table
            best_next_action = np.argmax(q_table[next_state])
            q_table[state, action] += alpha * (
                reward + gamma * q_table[next_state, best_next_action] - q_table[state, action]
            )

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # Aktualizacja epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Zapis wyników
        rewards.append(total_reward)
        epsilon_values.append(epsilon)
        steps_per_episode.append(steps)

    return q_table, rewards, steps_per_episode, epsilon_values

def plot_rewards_comparison(rewards_list, labels):
    plt.figure(figsize=(10, 6))

    for rewards, label in zip(rewards_list, labels):
        rolling_mean = np.convolve(rewards, np.ones(100) / 100, mode="valid")
        plt.plot(rolling_mean, label=label)

    plt.title("Porównanie średnich nagród w różnych konfiguracjach")
    plt.xlabel("Epizod")
    plt.ylabel("Średnia nagroda (ostatnie 100 epizodów)")
    plt.legend()
    plt.grid()
    plt.show()

def analyze_results(results):
    for name, rewards in results.items():
        print(f"\nStatystyki dla funkcji nagrody: {name}")
        print(f"  Średnia nagroda: {np.mean(rewards):.2f}")
        print(f"  Maksymalna nagroda: {np.max(rewards):.2f}")
        print(f"  Minimalna nagroda: {np.min(rewards):.2f}")

    # Wizualizacja wyników
    plot_rewards_comparison(list(results.values()), list(results.keys()))

# Modyfikacje funkcji nagrody
def default_reward(state, reward, done, action):
    return reward  # Domyślna nagroda

def penalize_illegal_moves(state, reward, done, action):
    if reward == -10:  # Kara za nielegalny ruch
        return -20
    return reward

def reward_quicker_completion(state, reward, done, action):
    if done and reward == 20:  # Zakończenie epizodu sukcesem
        return 50
    return reward

def reward_mix(state, reward, done, action):
    if reward == -10:  # Kara za nielegalny ruch
        return -30
    elif done and reward == 20:  # Zakończenie epizodu sukcesem
        return 100
    return reward

def plot_results(rewards, steps_per_episode, epsilon_values):
    plt.figure(figsize=(15, 10))

    # Wykres nagród
    plt.subplot(3, 1, 1)
    plt.plot(rewards, label="Nagroda")
    plt.title("Nagrody w kolejnych epizodach")
    plt.xlabel("Epizod")
    plt.ylabel("Nagroda")
    plt.grid()
    plt.legend()

    # Wykres liczby kroków
    plt.subplot(3, 1, 2)
    plt.plot(steps_per_episode, label="Liczba kroków", color="green")
    plt.title("Czas rozwiązania problemu (kroki w epizodzie)")
    plt.xlabel("Epizod")
    plt.ylabel("Liczba kroków")
    plt.grid()
    plt.legend()

    # Wykres wartości epsilon
    plt.subplot(3, 1, 3)
    plt.plot(epsilon_values, label="Epsilon", color="orange")
    plt.title("Zmiana epsilon w czasie")
    plt.xlabel("Epizod")
    plt.ylabel("Epsilon")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

def study_reward_functions():
    env = gym.make("Taxi-v3")
    reward_functions = [
        ("Domyślna funkcja nagrody", default_reward),
        ("Kara za nielegalne ruchy", penalize_illegal_moves),
        ("Premia za szybsze ukończenie", reward_quicker_completion),
        ("Mix", reward_mix),
    ]

    results = {}

    for label, reward_function in reward_functions:
        print(f"\nTrening z funkcją nagrody: {label}")
        wrapped_env = RewardWrapper(env, reward_function)
        q_table, rewards, steps_per_episode, epsilon_values = train_agent(
            wrapped_env, alpha=0.2, gamma=0.99, epsilon=1.0, epsilon_decay=0.996, epsilon_min=0.1,
            episodes=800, max_steps=250
        )
        results[label] = rewards 
        plot_results(rewards, steps_per_episode, epsilon_values)

    analyze_results(results)
    
if __name__ == "__main__":
    print("Badanie wpływu funkcji nagrody:")
    study_reward_functions()
