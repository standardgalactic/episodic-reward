import numpy as np
from visdom import Visdom
import csv

viz = Visdom()

win = None

data = np.array(0)

def update_viz(ep, ep_reward, algo):
    global win
    data.append(ep_reward)

    if win is None:
        win = viz.line(
            X=np.array([ep]),
            Y=np.array([ep_reward]),
            win=algo,
            opts=dict(
                title=algo,
                xlabel='Episodes',
                ylabel='Reward',
                fillarea=False
            )
        )
    else:
        viz.line(
            X=np.array([ep]),
            Y=np.array([ep_reward]),
            win=win,
            update='append',
            opts=dict(
                xlabel='Episodes',
                ylabel='Reward'
            )
        )

def write_reward_data(fname):
    with open('../data/' + fname, 'w') as f:
        f.write('Episodic Reward,\n')
        [f.write(str(d) + ',\n') for d in data]

