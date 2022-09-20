''' 
A basic agent (player) for the Starcraft II module.
Code based on: https://itnext.io/build-a-zerg-bot-with-pysc2-2-0-295375d2f58e

:author (Dr. Kevin Brewer)
:version (Summer 2019) (still valid Winter 2021)
:python (3.6) (and 3.8)
'''
# ------ import section ------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random as r
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time
from collections import deque

PLAYER_SELF = features.PlayerRelative.SELF
PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
PLAYER_ENEMY = features.PlayerRelative.ENEMY

#------- CONSTANTS ------
ACTION_STR = '_act'
BATCH_SIZE = 100
EXPLORATION_DECAY = 0.99
FUNCTIONS = actions.Functions
LEARNING_SIZE = 100
LOCATION_STR = '_loc'
MEMORY_SIZE = 1000000
RAW_FUNCTIONS = actions.Functions
FUNCTIONS = actions.FUNCTIONS

# ------ functions ------
def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))

# ------ agent class section ------
class ZergAgent(base_agent.BaseAgent):
    def __init__(self):
        '''
        The constructor method
        '''
        super(ZergAgent, self).__init__()

        self.memory = deque(maxlen = MEMORY_SIZE)
        self.exploration_rate = 1.0

        # load NN
        fn = input("Enter name of model to load (leave blank to start fresh): ")
        if fn != "":
            self.action_model = tf.keras.models.load_model(fn + ACTION_STR)
            self.location_model = tf.keras.models.load_model(fn + LOCATION_STR)
        else:
            self.observation_space = 7640
            self.action_space = 3
            self.location_space = 2
            # create the NN model
            initializer = tf.keras.initializers.GlorotNormal()
            self.action_model = tf.keras.models.Sequential(
                [
                    layers.Dense(
                        self.observation_space,
                        input_shape=(self.observation_space,),
                        activation="linear",
                        kernel_initializer = initializer,
                    ),
                    layers.Dense(
                        1024, activation= "linear", kernel_initializer = initializer
                    ),
                    layers.Dense(
                        1024, activation="linear", kernel_initializer = initializer
                    ),
                    layers.Dense(
                        self.action_space, activation="linear"
                        ),
                ]
            )

            self.action_model.compile(
                # loss=tf.nn.softmax_cross_entropy_with_logits(labels, logits),
                # loss=tf.nn.sigmoid_cross_entropy_with_logits(),
                loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(0.0001),
            )

            self.action_model.summary()

            self.location_model = tf.keras.models.Sequential(
                [
                    layers.Dense(
                        self.observation_space,
                        input_shape=(self.observation_space,),
                        activation="linear",
                        kernel_initializer = initializer,
                    ),
                    layers.Dense(
                        1024, activation= "linear", kernel_initializer = initializer
                    ),
                    layers.Dense(
                        1024, activation="linear", kernel_initializer = initializer
                    ),
                    layers.Dense(
                        self.location_space, activation="linear"
                        ),
                ]
            )

            self.location_model.compile(
                # loss=tf.nn.softmax_cross_entropy_with_logits(labels, logits),
                # loss=tf.nn.sigmoid_cross_entropy_with_logits(),
                loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(0.0001),
            )            
            
            self.location_model.summary()
    
    def unit_type_is_selected(self, obs, unit_type):
        '''
        A helper method to return True or False if selected unit(s) are of 
        type: unit_type

        Args:
            obs:         the observation object
            unit_type:   the desired unit type
        Return:
            True or False
        '''
        if (len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
            return True
        if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
            return True
        return False

    def get_units_by_type(self, obs, unit_type):
        '''
        A helper method to return existing unit(s) of 
        type: unit_type

        Args:
            obs:         the observation object
            unit_type:   the desired unit type
        Return:
            list of units
        '''
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]
  
    def can_do(self, obs, action):
        '''
        A helper method to if the action is available

        Args:
            obs:        the observation object
            action:     the action
        Return:
            True or False
        '''
        return action in obs.observation.available_actions

    def run_tests(self):
        self.exploration_rate = -1

    def step(self, obs):
        '''
        The method that handles each action decision. Only one action per
        step.

        Args:
            obs: the observation object
        Returns:
            The action (object) to be taken
        '''
        super(ZergAgent, self).step(obs)

        if (np.random.rand() <= self.exploration_rate):
            return self.scripted_step(obs)
        else:
            return self.NN_step(obs)

    #clear
    def NN_step(self, obs):
        # NN outputs softmax
        # outputs [no_op/0, select_army/7, move_screen/331]

        action = ""
        theState = self.get_state(obs)
        theState = np.array(theState).reshape(-1, self.observation_space)
        NN_output = self.action_model.predict(theState)
        currentAction = np.argmax(NN_output)
        print("NN output:", NN_output, "    action index: ", currentAction)
        if currentAction == 0:
            currentActionID = 0

        elif currentAction == 1:
            action = FUNCTIONS.select_army("select")
            currentActionID = 7
        else:
            loc = self.location_model.predict(theState)
            NNx = int(loc[0][0])
            NNy = int(loc[0][1])

            if NNx < 0 or NNx > 83:
                NNx = 0
            if NNy < 0 or NNy > 83:
                NNy = 0
            action = FUNCTIONS.Move_screen("now", [NNx, NNy])
            currentActionID = 331

        if currentActionID not in obs.observation.available_actions:
            action = FUNCTIONS.no_op()
            currentAction = 0
            currentActionID = 0
        print("Taking NN action with ID:", currentActionID)
        if currentActionID == 331:
            print ("location: ", NNx, NNy)

        return action

    #clear     
    def scripted_step(self, obs):
        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            beacon = _xy_locs(player_relative == PLAYER_NEUTRAL)
            if not beacon:
                return FUNCTIONS.no_op()
            beacon_center = np.mean(beacon, axis=0).round()
            return FUNCTIONS.Move_screen("now", beacon_center)
        else:
            return FUNCTIONS.select_army("select")
    
    #clear
    def get_state (self, obs):
        the_state = []
        #obs.observation.available_actions
        #obs.observation.feature_screen.player_relative
        #obs.observation["player"]

        avail_action_list = obs.observation.available_actions.tolist()
        #list of 6 possible actions
        # list of 16 actions
        # all possible is 0 through 52

        # player_id
        # minerals
        # vespene
        # food used
        # food cap
        # food used by army
        # food used by workers
        # idles worker count
        # army count
        # warp gate count
        # larva count
        
        #
        dummy = [0 for i in range(573)]
        for action in avail_action_list:
            dummy[action] = 1

        #player relative array
        dummy2 = [
            item for sublist in obs.observation.feature_screen.player_relative.tolist()
            for item in sublist
        ]

        dummy3 = obs.observation["player"].tolist()
        the_state = dummy + dummy2 + dummy3
        return the_state

    #clear
    def remember (self, curr_obs, action, reward, next_obs):
        state = self.get_state(curr_obs)
        next_state = self.get_state(next_obs)
        print("Length of state:", len(state))

        player_relative = curr_obs.observation.feature_screen.player_relative
        beacon = _xy_locs(player_relative == PLAYER_NEUTRAL)
        if not beacon:
            beaconLocation = [0,0]
        else:
            beaconLocation = np.mean(beacon, axis=0).round()

        action_code = int(str(FUNCTIONS[action[0]]).split("/")[0])
        if  action_code == 0:
            flatAction = [1.0,0.0,0.0]
        elif action_code == 7:
            flatAction = [0.0, 1.0, 0.0]
        elif action_code == 331:
            flatAction = [0.0, 0.0, 1.0]
        else:
            flatAction = [0.0, 0.0, 0.0]        

        self.memory.append([state, flatAction, beaconLocation, reward, next_state])

    def learn(self):
        # train the NN
        # don't do anything until you have enough data
        if len(self.memory) < LEARNING_SIZE:
            return
        if r.randint(0,100) < 95:
            return
        print("Learning!")
        # pick random data from all saved data to use to improve the model
        batch = r.sample(self.memory, LEARNING_SIZE)
        states_batch = []
        action_ys_batch = []
        location_ys_batch = []

        start_time = time.time()

        for s, fa, bl, reward, ns in batch:
            states_batch.append(s)
            action_ys_batch.append(fa)
            location_ys_batch.append(bl)

        # update the NN model
        states_batch = np.array(states_batch).reshape(-1, self.observation_space)
        location_ys_batch = np.array(location_ys_batch).reshape(-1, self.location_space)
        action_ys_batch = np.array(action_ys_batch).reshape(-1, self.action_space)

        print ("...prep time: --- %s seconds ---" % (time.time() - start_time))

        self.action_model.fit(
            states_batch, action_ys_batch, batch_size=BATCH_SIZE, epochs=20, verbose=0
        )
        
        self.location_model.fit(
            states_batch, location_ys_batch, batch_size=BATCH_SIZE, epochs=20, verbose=0
        )

        # update the exploration value
        self.exploration_rate *= EXPLORATION_DECAY

        print ("...total learning time: --- %s seconds ---" % (time.time() - start_time))

    #clear
    def status(self):
        print ("Agent:: Memory Length:", len(self.memory), " Exploration Rate: ", self.exploration_rate)

    def save(self):
        # save the NN
        fn = input("Enter model file name. Enter nothing to skip saving.:")
        if fn != "":
            self.action_model.save(fn + ACTION_STR)
            self.location_model.save(fn + LOCATION_STR)
        # for future loading use: self.model = tf.keras.models.load_model(fn)