from keras.layers import Dense, Input
from keras.models import  Model
from keras.optimizers import Adam
import  tensorflow as tf
import keras.backend as K

## Actor class that implements the online actor and it's target network and methods to get action predictions,
## train both the models and compute gradients using tensorflow.

class Actor:
    def __init__(self, env,sess, learning_rate, tau):
        self.sess = sess
        # set keras backend session as the tf session passed as an argument while instantiating an object
        K.set_session(sess)
        self.env = env

        # Model parameters
        self.learning_rate = learning_rate
        self.tau = tau

        # create online actor and it's target model
        self.model, self.state_input = self.create_model(self.env)
        self.target,_ = self.create_model(self.env)

        #Copy the weights of online actor to the target
        self.target.set_weights(self.model.get_weights())

        #Placeholder that will receieve dQ/dA from the critic network at each training iteration
        self.critic_grads = tf.placeholder(tf.float32,[None, self.env.action_space.shape[0]])

        #calculate actor gradients (dA/dTheta) and multiply by negative of dQ/dA to calculate the policy
        #gradient (dJ/dTheta = dA/dTheta * -dQ/dA)
        self.params_grads = tf.gradients(self.model.output, self.model.trainable_weights,-self.critic_grads)

        #zip the gradients with actor weights
        self.grads = zip(self.params_grads, self.model.trainable_weights)

        #Use the policy gradient calculated above to update the actor weights. Since we multiply negative of
        # dQ/dA, when we try to minimize J we are actually going uphill (gradient ascent)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients\
        (zip(self.params_grads, self.model.trainable_weights))

        #initialize all graph variables
        self.sess.run(tf.initialize_all_variables())

    #Same model architecture for both the actor and crtic. Important to note that the out layer has a
    # tanh activation to keep our output in the range.
    def create_model(self, env):
        state_input = Input(shape=env.observation_space.shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(env.action_space.shape[0], activation='tanh')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return model, state_input

    #Forward prop function. Input a state to output an action
    def predict_action(self, state):
        action = self.model.predict(state)
        return action

    #Same forward prop to get an action prediction from the target model. We input the next state here
    #The output from this function is used to calculate the TD target for the critic
    def predict_next_action(self, next_state):
        next_action = self.target.predict(next_state)
        return next_action

    #Actual train function which we call at each iteration. This function runs the optimize operation
    #to apply policy gradients to our actor network
    def train(self, batch, critic_grads):
        current_states,_,_,_,_,_ = batch
        self.sess.run(self.optimize, feed_dict={self.state_input: current_states,self.critic_grads: critic_grads})
        grads = self.sess.run(self.params_grads, feed_dict={self.state_input: current_states,self.critic_grads: critic_grads})
        return grads

    #Soft update for the target model to slowly track the actor weights. This is different from DQN
    # where we simply copy the online model's weights to the target after a certain number of iterations
    def update_target(self):
        actor_weights = self.model.get_weights()
        target_weights = self.target.get_weights()
        for i in range(len(actor_weights)):
            target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_weights[i]
        self.target.set_weights(target_weights)

    # functions to save actor model's architecture and weights.
    def save_model_architecture(self, file_name):
        model_json = self.model.to_json()
        with open(file_name, "w") as json_file:
            json_file.write(model_json)

    def save_weights(self, file_name):
        self.model.save_weights(file_name)



