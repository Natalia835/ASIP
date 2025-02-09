import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import Dense, Input
from keras.api.optimizers import Adam
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  
        self.epsilon = 1.0  
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.996
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()  
        self.update_target_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)[0]
            if done:
                target[action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
            states.append(state[0])
            targets.append(target)
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = gym.make('Taxi-v3')
state_size = 1
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

episodes = 500 
batch_size = 32
max_steps = 100

rewards = []
steps_per_episode = []
epsilon_values = [] 

for e in range(episodes):
    state, _ = env.reset()
    state = np.reshape([state], [1, state_size])
    total_reward = 0
    steps = 0
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
    
        if reward == -1:  
            reward = -0.1
        if done and reward == 20:  
            reward += 50
        elif done and reward == -10: 
            reward -= 20
        next_state = np.reshape([next_state], [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1
        if done:
            break
    rewards.append(total_reward)
    steps_per_episode.append(steps)
    epsilon_values.append(agent.epsilon) 
    print(f"Epizod {e+1}/{episodes}, Całkowita nagroda: {total_reward}, Kroki: {steps}, Wartość epsilon: {agent.epsilon:.2f}")
    agent.replay(batch_size)

    if e % 10 == 0:
        agent.update_target_model()

# Wykres 1: Całkowita nagroda w trakcie treningu
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.xlabel('Epizod')
plt.ylabel('Całkowita nagroda')
plt.title('Wyniki uczenia w trakcie epizodów')

# Wykres 2: Liczba kroków w każdym epizodzie
plt.subplot(1, 2, 2)
plt.plot(steps_per_episode)
plt.xlabel('Epizod')
plt.ylabel('Liczba kroków')
plt.title('Liczba kroków w każdym epizodzie')

plt.tight_layout()
plt.show()

# Wykres 3: Zmiana wartości epsilonu
plt.figure(figsize=(6, 6))
plt.plot(epsilon_values)
plt.xlabel('Epizod')
plt.ylabel('Wartość epsilon')
plt.title('Zmiana wartości epsilonu w czasie')
plt.show()

# Średnia nagroda i odchylenie standardowe (w ostatnich 50 epizodach)
if len(rewards) >= 50:
    mean_reward = np.mean(rewards[-50:])
    std_reward = np.std(rewards[-50:])
    print(f"Średnia nagroda (w ostatnich 50 epizodach): {mean_reward}")
    print(f"Odchylenie standardowe nagród (ostatnie 50 epizodów): {std_reward}")

# Średnia nagroda (pierwsze 50 epizodów i ostatnie 50 epizodów)
initial_avg_reward = np.mean(rewards[:50])
final_avg_reward = np.mean(rewards[-50:])
print(f"Średnia nagroda (pierwsze 50 epizodów): {initial_avg_reward}")
print(f"Średnia nagroda (ostatnie 50 epizodów): {final_avg_reward}")
