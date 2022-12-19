import gym
from gym import spaces
import numpy as np

class PortfolioChoice(gym.Env):

  def __init__(self, T=10):
    super(PortfolioChoice, self).__init__()
    # Actions are (risk_alloc, consumption)
    self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
    # Observations are (wealth, time)
    self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([np.inf, 1.0]), dtype=np.float64)

    self.curr_wealth = 1.0
    self.curr_time = 0
    self.horizon = T
    self.bequest = 0.1
    self.risk_aversion = 1 # log utility
    self.rho = 0.05

  # action space is [-1, 1]
  def rescale_riskalloc(self, risk_alloc):
    return (risk_alloc)*2+1
  def rescale_consumption(self, consumption):
    return (consumption+1)/2

  def update_wealth(self, action, time_step):
    time_scale = time_step*self.horizon # scaling returns based on time
    exp_return = 0.08
    rf_rate = 0.05
    sd_return = 0.15
    curr_wealth = self.curr_wealth
    risk_alloc = self.rescale_riskalloc(action[0])
    consumption = self.rescale_consumption(action[1])
    returns = (1-risk_alloc)*(np.exp(rf_rate*time_scale)-1) + risk_alloc*(np.exp((exp_return - sd_return**2/2)*time_scale + np.sqrt(time_scale)*np.random.normal(loc=0,scale=sd_return))-1)
    self.curr_wealth = curr_wealth*(1-consumption*time_scale)*(1 + returns)
  
  # consumption policy means fraction of current wealth consumed on an annualized basis 
  def calculate_utility(self, consumption):
    period = self.curr_time*self.horizon
    utility = np.log(1+consumption*1e6)*np.exp(-self.rho*period)
    return utility

  def step(self, action, time_step=0):
    # bit of a hack here so I can manually input timesteps
    if time_step==0:
      time_step = 1.0/self.horizon
    curr_time = self.curr_time

    # calculate reward
    consumption = self.rescale_consumption(action[1])*self.curr_wealth
    reward = self.calculate_utility(consumption)

    self.curr_time += time_step
    self.update_wealth(action, time_step)

    # check if end of simulation has been reached
    if round(self.curr_time,1)==1.0:
      done = True
    else:
      done = False

    if done==True and self.curr_wealth > 0:
      reward += self.bequest*self.calculate_utility(self.curr_wealth)

    if self.curr_wealth<=0:
      self.curr_wealth = 0
      done = True

    obs = np.array([self.curr_wealth, self.curr_time])
    info = {}

    return obs, reward, done, info

  def reset(self):
    self.curr_wealth = 1.0
    self.curr_time = 0
    return np.array([1.0, 0.0])
