import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# Parametry Q-learningu
alpha = 0.2
gamma = 0.985
epsilon = 1.0
epsilon_decay = 0.996
epsilon_min = 0.1
episodes = 800
max_steps = 250

# Polityka epsilon-greedy
def choose_action(state, q_table, epsilon, action_space_size):
    if np.random.rand() < epsilon:
        return np.random.randint(action_space_size)  
    else:
        return np.argmax(q_table[state])             

def train_agent(alpha, gamma, epsilon, epsilon_decay, epsilon_min, episodes, max_steps):
    env = gym.make("Taxi-v3", render_mode=None)
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n

    q_table = np.zeros((state_space_size, action_space_size))
    rewards = []
    steps_per_episode = []
    epsilon_values = []

    start_time = time.time()

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
        steps_per_episode.append(steps)
        epsilon_values.append(epsilon)

    training_time = time.time() - start_time

    return q_table, rewards, steps_per_episode, epsilon_values, training_time

def test_agent(q_table, max_steps=100):
    env = gym.make("Taxi-v3", render_mode="human")
    state, info = env.reset()
    total_reward = 0
    steps = 0

    for step in range(max_steps):
        action = np.argmax(q_table[state])
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1
        if done:
            break

    print(f"Nagroda uzyskana podczas testu: {total_reward}")
    print(f"Liczba kroków w epizodzie testowym: {steps}")
    env.close()
    return total_reward, steps

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

# Ocenienie stabilności i szybkości uczenia
def evaluate_learning(rewards):
    rolling_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
    stability = np.std(rolling_avg[-50:])  # Odchylenie standardowe nagród na końcu
    print(f"Stabilność (odchylenie standardowe ostatnich 50 epizodów): {stability:.2f}")

    # Szybkość uczenia (średnia nagroda na początku i końcu)
    start_avg = np.mean(rewards[:100])
    end_avg = np.mean(rewards[-100:])
    print(f"Średnia nagroda na początku (pierwsze 100 epizodów): {start_avg:.2f}")
    print(f"Średnia nagroda na końcu (ostatnie 100 epizodów): {end_avg:.2f}")

# Badanie wpływu parametrów
def study_hyperparameters():
    epsilon_decays = [0.9, 0.99, 0.999]

    for epsilon_decay in epsilon_decays:
        print(f"\nTrening agenta: alpha={alpha}, gamma={gamma}, wsp. zmniejszenia ekploracji: {epsilon_decay}")
        q_table, rewards, steps_per_episode, epsilon_values, training_time = train_agent(
            alpha, gamma, epsilon, epsilon_decay, epsilon_min, episodes, max_steps
        )
        plot_results(rewards, steps_per_episode, epsilon_values)
        print(f"Czas trwania treningu: {training_time:.2f} sekund")

if __name__ == "__main__":

    print("Trening agenta Q-learning...")
    q_table, rewards, steps_per_episode, epsilon_values, training_time = train_agent(
        alpha, gamma, epsilon, epsilon_decay, epsilon_min, episodes, max_steps
    )
    print(f"Czas trwania treningu: {training_time:.2f} sekund")

    plot_results(rewards, steps_per_episode, epsilon_values)

    print("\nAnaliza jakości uczenia:")
    evaluate_learning(rewards)

    print("\nTestowanie wyuczonego agenta:")
    test_agent(q_table)

    print("\nBadanie wpływu hiperparametrów:")
    study_hyperparameters()
