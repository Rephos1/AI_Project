import numpy as np


'''
Leduc Rules:
2 rounds, 2-bet limit
2 suits * 3 cards per = 6 cards
1st round: 1 chip ante +  single card is dealt, chip bet = 2
2nd round: 1 board card revealed, another betting round, chip bet is 4 

'''
ACTIONS = 3 # Check/Fold  Bet/Call Reraise


class Node:
  def __init__(self,num_actions):
    self.num_actions = ACTIONS
    self.regret_sum = np.zeros(num_actions)
    self.strategy = np.zeros(num_actions)
    self.strategy_sum = np.zeros(num_actions)
    self.num_actions = np.zeros(num_actions)
  
  def get_strategy(self, reaching_prob):
    normalizing_sum = 0
    for a in range(self.num_actions):
      if self.regret_sum[a]>0:
        self.strategy[a] = self.regret_sum[a]
      else:
        self.strategy[a] = 0
      normalizing_sum +=self.strategy[a]
    
    for a in range(self.num_actions):
      if normalizing_sum > 0:
        self.strategy[a] /= normalizing_sum
      else:
        self.strategy[a] = 1.0 / self.num_actions
    return self.strategy
  
  def get_average_strategy(self):
      
    





class LeducCFR:
  def __init__(self, iterations, decksize = 6):
    self.iterations = iterations
    self.decksize = decksize
    self.cards = np.arange(decksize)
    
