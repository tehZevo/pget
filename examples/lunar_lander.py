import tensorflow as tf
import numpy as np
import gym

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from pget import Agent

tf.enable_eager_execution()

model = Sequential([
  Dense(32, input_shape=[8], activation="relu"),
  Dense(32, activation="relu"),
  Dense(4, activation="softmax"),
])

agent = Agent(model, action_type="discrete", lr=1e-3)
agent.model.summary()

env = gym.make("LunarLander-v2")
action_repeat = 4

s = env.reset()

while True:
  a = agent.get_action(s)
  a = np.argmax(a)

  r = 0
  for i in range(action_repeat):
    s, step_r, done, info = env.step(a)
    r += step_r
    if done:
      break

  agent.train(r)
  env.render()

  if done:
    s = env.reset()
