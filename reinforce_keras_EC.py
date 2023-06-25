import tensorflow as tf
from network_EC import PolicyGradientNetwork
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys
from copy import copy, deepcopy

class Agent:
    
    def __init__(self, alpha=0.0001/np.sqrt(10), gamma=0.95, n_actions=4,
                 layer1_size=300, layer2_size=300, lambda_entr=5e-3):

        self.kappa = 0.9
        self.lambda_pol = 4.0
        self.lambda_entr = lambda_entr
        self.gamma = gamma
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward1_memory = []
        self.reward2_memory = []
        self.mean_returns = []
        self.policy = PolicyGradientNetwork(n_actions, layer1_size, layer2_size)
        self.policy.compile(optimizer=Adam(learning_rate=alpha, beta_1=0.9, beta_2=0.999))
        self.number_of_epochs_trained = 0
        
    def load_policy(self, filename):
        self.policy = tf.keras.models.load_model(filename)

    def choose_action(self, state):
        state_tf = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.policy(state_tf)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()

        return action.numpy()[0]
    
    def choose_action_batch(self, state):
        state_tf = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.policy(state_tf)
        print('here')
        print('probs',probs)
        print("sum",tf.reduce_sum(probs))
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()
        print(action)

        return action.numpy()[0]

    def store_transition(self, observation, action, reward1, reward2):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward1_memory.append(reward1)
        self.reward2_memory.append(reward2)
        
    def store_batch(self, states, actions, rewards1, rewards2):
        self.state_memory = states
        self.action_memory = actions
        self.reward1_memory = rewards1
        self.reward2_memory = rewards2
        

    def learn(self):
        actions = np.array(self.action_memory)
        rewards1 = np.array(self.reward1_memory)
        rewards2 = np.array(self.reward2_memory)
        
        batch_size = np.shape(rewards1)[0]
        N_gates = np.shape(rewards1)[1]
        current_epoch = len(self.mean_returns)

        G = np.zeros_like(rewards1)
        for i in range(batch_size):
            
            for t in range( N_gates ):
                G_sum = 0
                for k in range(N_gates-t):
                    G_sum += rewards1[i,t+k] * self.gamma**k
                G[i,t] = (1-self.gamma)*G_sum + rewards2[i,t]
        
        self.mean_returns.append( G[:,:].mean(axis=0) )
        
        b = np.ones( N_gates )
        b *= (1-self.kappa)
        
        for t in range(N_gates):
            factor = 0
            if (current_epoch>0): 
                for n in range(current_epoch):
                    factor += self.kappa**n * self.mean_returns[current_epoch-1-n][t]
                b[t] *= factor
                
        G_minus_b = tf.convert_to_tensor( G - b )
        
        with tf.GradientTape() as tape:
            loss=0.
            for i in range(batch_size):
                
                states_tf = tf.convert_to_tensor(self.state_memory[i])
                actions_tf = tf.convert_to_tensor(self.action_memory[i])
                probs = self.policy(states_tf)
                log_probs = tf.math.log( probs )
                slice_indices = tf.transpose( tf.stack( (tf.range(0,N_gates), actions_tf) ) )
                log_probs_a = tf.gather_nd( log_probs, slice_indices )
                
                probs_log_probs = probs * log_probs
                sum_over_s_and_a = tf.reduce_sum( probs_log_probs )
                
                loss += - ( self.lambda_pol * tf.reduce_sum( G[i,:] * log_probs_a ) \
                                  - self.lambda_entr * sum_over_s_and_a/N_gates )
                
                    
            loss /= batch_size
                
        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))

        self.state_memory = []
        self.action_memory = []
        self.reward1_memory = []
        self.reward2_memory = []
        self.number_of_epochs_trained += 1
        
        