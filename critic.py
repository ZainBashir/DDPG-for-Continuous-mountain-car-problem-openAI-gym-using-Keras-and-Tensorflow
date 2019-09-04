from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.merge import Add

import tensorflow as tf
import keras.backend as K

##Critic class that implements the critic online and target models and provides methods to make
##Q-value predictions, train the model and calculate gradients (change in Q value w.r.t to the change
# in actions)
class Critic:
    def __init__(self,env,sess, learning_rate, tau, discount_factor):
       self.sess = sess
       # set keras backend session as the tf session passed as an argument while instantiating an object
       K.set_session(sess)
       self.env = env

       #model parameters
       self.gamma = discount_factor
       self.tau = tau
       self.learning_rate = learning_rate
       self.model, self.state_input, self.action_input = self.create_model(self.env)

       #Create online critic and it's target model
       self.target, _,_ = self.create_model(self.env)
       self.target.set_weights(self.model.get_weights())

       #Define an operation to calculate the gradients (dQ/dA). This operation will be run every training
       #iteration where we get new actions from the actor
       self.gradients = tf.gradients(self.model.output, self.action_input)

       #Initialize all graph variables
       self.sess.run(tf.initialize_all_variables())

    #Architecture of the critic model and it's target model. The critic take in two inputs; a state and
    #the action taken in that state to calculate the Q-value of a state action pair.
    #The action input is passes through one dense layer before being merged with the main network.
    def create_model(self,env):
        state_input = Input(shape=env.observation_space.shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=env.action_space.shape)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return model, state_input, action_input

    # Forward pass through the network to calculate the Q-value of being in the current state and an action
    #taken
    def predict_qvalue(self,state,action):
        qvalue = self.model.predict([state, action])
        return qvalue

    # Forward pass through the network to calculate the Q-value of being in the next state and
    # the next action taken
    def predict_next_qvalue(self,next_state, next_action):
        next_qvalue = self.target.predict([next_state, next_action])
        return next_qvalue

    #Train function that updates the parameters of the critic model based on the bellman loss
    def train(self, batch):
        current_states, actions, rewards, next_states, next_actions, dones = batch

        next_qvalues = self.predict_next_qvalue(next_states, next_actions)
        next_qvalues[dones] = 0

        td_targets = rewards + self.gamma*next_qvalues

        self.model.fit([current_states,actions], td_targets, verbose=0)

    #Function to calculate the gradient dQ/dA which will be used to calculate the policy gradient
    def calc_grads(self,batch):
        current_states,actions,_,_,_,_ = batch
        grads = self.sess.run(self.gradients, feed_dict={
            self.state_input: current_states,
            self.action_input: actions
        })[0]

        return grads

    #Soft updates for the target model
    def update_target(self):
        critic_weights = self.model.get_weights()
        target_weights = self.target.get_weights()
        for i in range(len(critic_weights)):
            target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * target_weights[i]
        self.target.set_weights(target_weights)

    #Saving model architecture and weights
    def save_model_architecture(self, file_name):
        model_json = self.model.to_json()
        with open(file_name, "w") as json_file:
            json_file.write(model_json)

    def save_weights(self, file_name):
        self.model.save_weights(file_name)
