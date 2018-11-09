import numpy as np
import gym
from IPython.display import HTML
from PIL import Image,ImageDraw, ImageFont
from sklearn.preprocessing import PolynomialFeatures

class LinearAgent:
    def __init__(self,
            weights=None,
            order=1,
            id=None,
            pedigree=None,
            game="CartPole-v1"
        ):
        # The order of the linear agent
        self.order = order

        # The game
        self.game = game

        # I hardcode the observation and action spaces here
        if game == "CartPole-v1":
            self.obs = 4
            self.output = 1
            #self.ws = 1+order*self.obs
        elif game == "MountainCar-v0":
            self.obs = 2
            self.output = 3
            #self.ws = self.output*(1+order*self.obs)
        elif game == "Pendulum-v0":
            self.game = "Pendulum-v0"
            self.obs = 3
            self.output = 5
            #self.ws = self.output*(1+order*self.obs)
        elif game == "LunarLander-v2":
            self.game = "LunarLander-v2"
            self.obs = 8
            self.output = 4
            #self.ws = self.output*(1+order*self.obs)
        else:
            self.game = "BipedalWalker-v2"
            self.obs = 24
            self.output = 20

        self.poly = PolynomialFeatures(degree=order)
        self.ws = (self.poly.fit_transform(np.zeros((1, self.obs))).shape[1], self.output)

        # This segment handles inheriting weights and pedigree
        if weights is None:
            self.w = np.random.uniform(-1, 1, self.ws)
        else:
            self.w = weights
        if id is not None:
            self.pedigree = self.ws[0] * self.ws[1] * [str(id)]
        elif pedigree is not None:
            self.pedigree = pedigree
        else:
            self.pedigree = self.ws[0] * self.ws[1] * [str(-1)]

    def predict(self, x):
        x = x.reshape(1, -1)

        Y = np.dot(self.poly.transform(x), self.w)

        if self.game == "CartPole-v1": # sigmoid
            z = Y[0]
            a = 1 / (1 + np.exp(-z))
            return int(np.round(a))
        elif self.game in ["MountainCar-v0", "LunarLander-v2"]: # cheap softmax
            return int(np.argmax(Y, axis=1))
        elif self.game == "Pendulum-v0":
            return [np.argmax(Y, axis=1) - 2]
        else:
            Y = Y.flatten()
            hip1 = float(np.argmax(Y[0:4]))/2 -1
            knee1 = float(np.argmax(Y[5:9]))/2 -1
            hip2 = float(np.argmax(Y[10:14]))/2 -1
            knee2 = float(np.argmax(Y[15:19]))/2 -1
            return [hip1, knee1, hip2, knee2]


    def render(self, filename, episodes=5, limit=1000):
        # Prepend the sub-folder
        filename = "./images/"+filename

        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 32)

        env = gym.make(self.game)

        frames = []
        for e in range(episodes):
            observation = env.reset()
            total_reward = 0
            for t in range(limit):
                # Skip some frames
                if t % 5 == 0:
                    frames.append(Image.fromarray(env.render(mode = 'rgb_array'), "RGB"))
                    # Draw the episode number
                    ImageDraw.Draw(frames[-1]).text((0,frames[-1].size[1]-64), f"episode  {e+1}",
                        font=fnt, fill=(0,0,0,0))
                    # Draw the timestep number
                    ImageDraw.Draw(frames[-1]).text((0,frames[-1].size[1]-32), f"timestep {t}",
                        font=fnt, fill=(0,0,0,0))
                    # Draw the total reward
                    ImageDraw.Draw(frames[-1]).text((frames[-1].size[0]-128,frames[-1].size[1]-32),
                        str(int(total_reward)), font=fnt, fill=(0,0,0,0))
                action = self.predict(observation)
                observation, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
            # Triple the last frame
            frames.append(frames[-1])
            frames.append(frames[-1])
        env.close()

        # Resize the frames
        for f in frames:
            f.thumbnail(size=[300,200])

        # Save the GIF
        frames[0].save(filename, format='GIF', append_images=frames[1:], save_all=True, duration=75, loop=0)

        # Return the filename to be input in HTML()
        return(filename)
