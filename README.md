# DDPG-for-Continuous-mountain-car-problem-openAI-gym-using-Keras-and-Tensorflow

Deep deterministic policy gradient using Keras and Tensorflow with python to solve the Continous mountain car problem provided
by OpenAI gym.

Input to the model is the position and velocity information of the car while the output is a single real-valued number indicating
the deterministic action to take given a state.

This work uses many tuorials that I found online and much of my code is similar to the already existing tutorials available online


## How to use

1) DDPG.py is the file containing the main training loop. Run this for training
2) The files actor.py, critic.py, replay_memory.py, ounoise.py are the different classes used in the model.
3) evaluatemodel.py is the python script to evaluate your trained actor's performance
4) The folder Model architecture and trained weights contains the actor's architecture as a json file and the trained weights after 750 episodes of 500 iterations each. Give the path of the these two files in the evaluatemodel.py script to observe the performance of my training.
5) A reward of 93 is obtained consistently with my trained model.
