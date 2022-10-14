
import random, numpy, math, gym
import time
NAME = "SARSA-{}".format(int(time.time()))

# -------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras import backend as K

import time
NAME = "SARSA-{}".format(int(time.time()))

DAY0 = 50
DAYN = 60
REWARDS = {}

for i in range(0,DAYN,1):
    REWARDS[i]=[]

from tensorflow.keras.optimizers import RMSprop

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        #self.tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
        # self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):
        l_input = Input(batch_shape=(None, self.stateCnt))
        l_input1 = Lambda(lambda x: x[:, 0:self.stateCnt - 7])(l_input)
        l_input2 = Lambda(lambda x: x[:, -7:])(l_input)
        l_input1 = Reshape((DEFAULT_NUM_TCLS, 1))(l_input1)
        l_Pool = AveragePooling1D(pool_size=self.stateCnt - 7)(l_input1)
        l_Pool = Reshape([1])(l_Pool)
        l_dense = Concatenate()([l_Pool, l_input2])
        l_dense = Dense(100, activation='relu')(l_dense)
        l_dense = Dropout(0.3)(l_dense)
        out_value = Dense(80, activation='linear')(l_dense)
        # model = Model(inputs=l_input, outputs=[out_tcl_actions,out_price_actions,out_deficiency_actions,out_excess_actions, out_value])
        model = Model(inputs=l_input, outputs=out_value)
        model.make_predict_function()
        opt = RMSprop(lr=0.00025)

        model.compile(loss='mse', optimizer=opt)
        return model


    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=100, epochs=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()


class Memory:  # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def extra(self, next_a):
        if len(self.samples) > 1:
            self.samples[-2][-1] = next_a

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)


# -------------------- AGENT ---------------------------
# hyperparameters
MEMORY_CAPACITY = 500
BATCH_SIZE = 300

GAMMA = 1.0

MAX_EPSILON = 0.5
MIN_EPSILON = 0.004
LAMBDA = 0.00001  # speed of decay


# MAX_EPSILON = 0.4
# MIN_EPSILON = 0.004
# LAMBDA =5e-5  # speed of decay


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s, deter):
        if deter==True:
            return numpy.argmax(self.brain.predictOne(s))
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)

        return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)
        self.memory.extra(sample[1])
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = max(MAX_EPSILON -LAMBDA * self.steps, MIN_EPSILON)

    def replay(self):
        if len(self.memory.samples)<2:
            return
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]
            a_ = o[4]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * p_[i][a_]

            x[i] = s
            y[i] = t

        self.brain.train(x, y)


# -------------------- ENVIRONMENT ---------------------
from tcl_env_dqn_1 import *

class Environment:
    def __init__(self, render = False, **kwargs):
        self.env = MicroGridEnv(**kwargs)
        self.render=render
        self.run_count = 0
        self.R_hist = []

    def run(self, agent, day=None):
        s = self.env.reset(day=day)
        R = 0
        while True:

            a = agent.act(s, deter=self.render)

            s_, r, done, info = self.env.step(a)
            if self.render: self.env.render('SARSA')

            if done:  # terminal state
                s_ = None

            agent.observe([s, a, r, s_,None])


            s = s_
            R += r
            #wandb.log({"r":r})

            if done:

                agent.replay()
                break

        REWARDS[self.env.day].append(R)
        self.R_hist.append(R)
        self.run_count += 1
        print("Day ", self.env.day)
        print("R= ", R)



# -------------------- MAIN ----------------------------
from datetime import datetime
if __name__=="__main__":
  # PROBLEM = TCLEnv
  env = Environment()#day0 = DAY0, dayn=DAYN)

  stateCnt = env.env.observation_space.shape[0]
  actionCnt = env.env.action_space.n

  agent = Agent(stateCnt, actionCnt)


  import time
  d = datetime.now()

  t0=time.time()
  for _ in range(1000):
      env.run(agent, day=DAY0)
  output_data = pd.read_csv('training_reward.csv')
  output_data[f"Reward SARSA {d.day}/{d.month}, {d.hour}:{d.minute}:{d.second}"] = env.R_hist
  output_data.to_csv("training_reward.csv",index=False)
  

  print('training_time:', time.time()-t0)

  env_test = Environment(render=True)
  for day in range(DAY0,DAYN):
    env_test.run(agent,day=day)
  
  output_data = pd.read_csv('test_reward.csv')
  output_data[f"Reward SARSA {d.day}/{d.month}, {d.hour}:{d.minute}:{d.second}"] = env_test.R_hist
  output_data.to_csv("test_reward.csv",index=False)