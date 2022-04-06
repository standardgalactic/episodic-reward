import gym
import os

import ray
from ray import tune

from ray.rllib.agents.dqn import DQNTrainer


config = {
    "env": "SpaceInvaders-v0",
    "num_workers": 1,
    "framework": "torch",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [32, 64, 512],
        "fcnet_activation": "relu",
        "grayscale": True,

    },
    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    }
}


# Create our RLlib Trainer.
trainer = DQNTrainer(config=config)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for _ in range(3):
    print(trainer.train())

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
trainer.evaluate()