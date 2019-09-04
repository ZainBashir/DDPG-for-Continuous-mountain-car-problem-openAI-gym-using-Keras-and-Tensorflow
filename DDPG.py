import gym
from replay_memory import ReplayMemory
from actor import Actor
from critic import Critic
from ounoise import OUNoise

import tensorflow as tf
import keras.backend as K
import pickle


def main():
    sess = tf.Session()
    K.set_session(sess)
    env = gym.make("MountainCarContinuous-v0")

    #Parameters
    memory_size = 100000
    batch_size = 32
    tau = 0.001
    lr_actor = 0.0001
    lr_critic = 0.001
    discount_factor = 0.99
    episodes = 1001
    time_steps = 501
    collect_experience = 50000
    save_frequency = 250
    ep_reward = []
    training = False

    #Noise objecct
    noise = OUNoise(env.action_space)
    #Initialize actor and critic objects
    actor = Actor(env,sess, lr_actor, tau)

    #Uncomment to the following line to save the actor model architecture as json file. Need to be saved
    #once only

    # actor.save_model_architecture("Actor_model_architecture.json")
    critic = Critic(env,sess, lr_critic, tau, discount_factor)

    #Initialize replay memory of size defined by memory_size
    replay_memory = ReplayMemory(memory_size)

    #Toggle between true and false for debugging purposes. For training it is always true
    run = True
    if run:
        #Loop over the number of episodes. At eqach new episode reset the environment, reset the noise
        #state and set total episode reward to 0
        for episode in range (episodes):
            state = env.reset()
            noise.reset()
            episode_reward = 0

            #Loop over the number of steps in an episode
            for time in range (time_steps):
                #Uncomment the following line of you want to visualize the mountain car during training.
                #Can also be trained without visualization for the case where we are using
                #position and velocities as state variables.

                # env.render()

                #Predict an action from the actor model using the current state
                action = actor.predict_action(state.reshape((1,2)))[0]

                #Add ohlnbeck noise to the predicted action to encourage exploration of the environment
                exploratory_action = noise.get_action(action,time)

                #Take the noisy action to enter the next state
                next_state ,reward, done, _ = env.step(exploratory_action)

                #Predict the action to be taken given the next_state. This next state action is predicted
                #using the actor's target model
                next_action = actor.predict_next_action(next_state.reshape((1,2)))[0]

                #Append this experience sample to the replay memory
                replay_memory.append(state,exploratory_action,reward,next_state, next_action,done)

                #Only start training when there are a minimum number of experience samples available in
                #memory
                if replay_memory.count() == collect_experience:
                    training = True
                    print('Start training')

                #When training:
                if training:
                    # 1)first draw a random batch of samples from the replay memory
                    batch = replay_memory.sample(batch_size)
                    # 2) using this sample calculate dQ/dA from the critic model
                    grads = critic.calc_grads(batch)
                    # 3) calculate dA/dTheta from the actor using the same batch
                    # 4) multiply dA/dTheta by negative dQ/dA to get dJ/dTheta
                    # 5) Update actor weights such that dJ/dTheta is maximized
                    # 6) The above operation is easily performed by minimizing the value obtained in (4)
                    t_grads = actor.train(batch,grads)

                    # update critic weights by minimizing the bellman loss. Use actor target to compute
                    # next action in the next state (already computed and stored in replay memory)
                    # in order to compute TD target
                    critic.train(batch)

                    #After each weight update of the actor and critic online model perform soft updates
                    # of their targets so that they can smoothly and slowly track the online model's
                    #weights
                    actor.update_target()
                    critic.update_target()

                #Add each step reward to the episode reward
                episode_reward += reward

                #Set current state as next state
                state = next_state

                #If target reached before the max allowed time steps, break the inner for loop
                if done:
                    break

            #Store episode reward
            ep_reward.append([episode, episode_reward])

            #Print info for each episode to track training progress
            print("Completed in {} steps.... episode: {}/{}, episode reward: {} "
                  .format(time, episode, episodes, episode_reward))

            #Save model's weights and episode rewards after each save_frequency episode
            if training and (episode % save_frequency) == 0:
                print('Data saved at epsisode:', episode)
                actor.save_weights('./Model/DDPG_actor_model_{}.h5'.format(episode))
                pickle.dump(ep_reward, open('./Rewards/rewards_{}.dump'.format(episode), 'wb'))

        # Close the mountain car environment
        env.close()


########## RUN
if __name__ == "__main__":
    main()