import gym
import tensorflow as tf
import keras.backend as K
from keras.models import model_from_json


def main():
    sess = tf.Session()
    K.set_session(sess)
    env = gym.make("MountainCarContinuous-v0")

    episodes = 400
    time_steps = 501


    # load json and create model
    json_file = open('./Model/Actor_model_architecture.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    actor = model_from_json(loaded_model_json)
    actor.load_weights("./Model/DDPG_actor_model_750.h5")

    run = True
    if run:
        for episode in range (episodes):
            state = env.reset()
            episode_reward = 0

            for time in range (time_steps):
                env.render()
                action = actor.predict(state.reshape((1,2)))[0]



                # print("deterministic action:",action)
                # print("noisy action:", exploratory_action)

                next_state ,reward, done, _ = env.step(action)

                episode_reward += reward
                state = next_state


                if done:
                    break
            print("Completed in {} steps.... episode: {}/{}, episode reward: {} "
                  .format(time, episode, episodes, episode_reward))
        env.close()



if __name__ == "__main__":
    main()