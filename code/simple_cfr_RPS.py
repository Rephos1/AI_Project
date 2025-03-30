import numpy as np

ROCK = 0
PAPER = 1
SCISSORS = 2
NUM_OF_ACTIONS = 3
                            #Rock   #Paper  #Scissors
opponent_strategy = np.array([0.3,  0.3,    0.4])

# Payoff table for Rock Paper Scissors
payoff_table = [
    [0, -1, 1],   # Rock
    [1, 0, -1],   # Paper
    [-1, 1, 0]]   # Scissors

def get_strategy(regretSum: np.array, strategy_sum):
    strategy = np.maximum(regretSum, 0) # regretSum[i] =  0 if i < 0
    normalizing_sum = np.sum(strategy)        # sum of all
    if normalizing_sum > 0:
        strategy /= normalizing_sum
    else:
        strategy = np.full(NUM_OF_ACTIONS, 1.0 / NUM_OF_ACTIONS) # uniform distribution
    strategy_sum += strategy
    return strategy, strategy_sum

def get_action(strategy):
    return np.random.choice(NUM_OF_ACTIONS, p=strategy)

def train(iterations=1000):
    regretSum = np.zeros(NUM_OF_ACTIONS)
    strategy_sum = np.zeros(NUM_OF_ACTIONS )
    action_utility = np.zeros(NUM_OF_ACTIONS)


    for _ in range(iterations):
        current_strategy, strategy_sum = get_strategy(regretSum, strategy_sum)
        my_action = get_action(current_strategy)
        other_action = get_action(opponent_strategy)

        my_payoff = payoff_table[my_action][other_action]

        for a in range(NUM_OF_ACTIONS):
            action_utility[a] = payoff_table[a][other_action]
            regretSum[a] += action_utility[a] - my_payoff
            print(current_strategy, strategy_sum,other_action)


    return strategy_sum / iterations



result = train(10)



print(f'Result: {result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f} ')

