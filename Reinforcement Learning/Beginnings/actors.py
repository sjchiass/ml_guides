import numpy as np
import gym
from IPython.display import HTML
import PIL.Image

class SimpleActor:
    def __init__(self, weights=None):
        if weights is None:
            self.w = np.random.uniform(-1, 1, 5)
        else:
            self.w = weights
    def predict(self, x):
        # First-order formula
        z = self.w[0] + self.w[1]*x[0] + self.w[2]*x[1] + self.w[3]*x[2] + self.w[4]*x[3]

        # Sigmoid
        a = 1 / (1 + np.exp(-z))
        return int(round(a))
    def render(self, filename):
        env = gym.make('CartPole-v0')

        observation = env.reset()
        frames = []
        for t in range(1000):
            frames.append(PIL.Image.fromarray(env.render(mode = 'rgb_array'), "RGB"))
            action = self.predict(observation)
            observation, reward, done, info = env.step(action)
            if done:
                break
        env.close()

        # Save the GIF
        frames[0].save(filename, format='GIF', append_images=frames[1:], save_all=True, duration=5, loop=0)

        # Return the filename to be input in HTML()
        return(filename)


class ComplexActor(SimpleActor):
    def __init__(self, weights=None):
        if weights is None:
            self.w = np.random.uniform(-1, 1, 9)
        else:
            self.w = weights
    def predict(self, x):
        # First-order part
        z = self.w[0] + self.w[1]*x[0] + self.w[2]*x[1] + self.w[3]*x[2] + self.w[4]*x[3]
                
        # Second-order part 
        z += self.w[5]*x[0]**2 + self.w[6]*x[1]**2 + self.w[7]*x[2]**2 + self.w[8]*x[3]**2

        # Sigmoid
        a = 1 / (1 + np.exp(-z))
        return int(round(a))