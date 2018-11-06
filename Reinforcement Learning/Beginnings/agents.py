import numpy as np
import gym
from IPython.display import HTML
import PIL.Image

class LinearAgent:
    def __init__(self, 
            weights=None, 
            order=1, 
            id=None, 
            pedigree=None, 
            cartpole=True
        ):
        # The order of the linear agent
        self.order = order

        # I hardcode the observation and action spaces here
        if cartpole:
            self.game = "CartPole-v1"
            self.obs = 4
            self.ws = 1+order*self.obs
            self.output = 1
        else:
            self.game = "MountainCar-v0"
            self.obs = 2
            self.output = 3
            self.ws = self.output*(1+order*self.obs)

        # This segment handles inheriting weights and pedigree
        if weights is None:
            self.w = np.random.uniform(-1, 1, self.ws)
        else:
            self.w = weights
        if id is not None:
            self.pedigree = self.ws * [str(id)]
        elif pedigree is not None:
            self.pedigree = pedigree
        else:
            self.pedigree = self.ws * [str(-1)]
        
    def predict(self, x):
        Y = []

        for i in range(self.output):
            # Intercept
            z = self.w[i*self.obs]

            # Orders
            for j in range(self.order):
                for o in range(self.obs):
                    z += self.w[i*self.obs+1+j+o]*x[o]**(j+1)
            Y.append(z)

        if self.game == "CartPole-v1": # sigmoid
            z = Y[0]
            a = 1 / (1 + np.exp(-z))
            return int(round(a))        
        else: # softmax
            z_num = sum([np.exp(i) for i in Y])
            a = [np.exp(i)/z_num for i in Y]
            return np.argmin(a)

    def render(self, filename, episodes=5, limit=1000):
        # Prepend the sub-folder
        filename = "./images/"+filename

        env = gym.make(self.game)

        frames = []
        for e in range(episodes):
            observation = env.reset()
            for t in range(limit):
                # Skip some frames
                if t % 10 == 0:
                    frames.append(PIL.Image.fromarray(env.render(mode = 'rgb_array'), "RGB"))
                action = self.predict(observation)
                observation, reward, done, info = env.step(action)
                if done:
                    break
        env.close()

        # Resize the frames
        for f in frames:
            f.thumbnail(size=[300,200])

        # Save the GIF
        frames[0].save(filename, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)

        # Return the filename to be input in HTML()
        return(filename)
