import numpy as np
import random
import sys

# sys.setrecursionlimit(100000)  

import time

start_time = time.time()

class Node:
    def __init__(self, num_actions):
        self.regret_sum = np.zeros(num_actions)
        self.strategy = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.num_actions = num_actions

        #Strategy =  normalised regrets
    def get_strategy(self, reach_prob=1):
        normalizing_sum = 0
        for a in range(self.num_actions):
            if self.regret_sum[a] > 0:
                self.strategy[a] = self.regret_sum[a]
            else:
                self.strategy[a] = 0
            normalizing_sum += self.strategy[a]

        for a in range(self.num_actions):
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 1.0 / self.num_actions

        return self.strategy

    def get_average_strategy(self):
        avg_strategy = np.zeros(self.num_actions)
        normalizing_sum = 0

        for a in range(self.num_actions):
            normalizing_sum += self.strategy_sum[a]
        for a in range(self.num_actions):
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / self.num_actions

        return avg_strategy


class KuhnCFR:
    def __init__(self, iterations, decksize):
        self.iterations = iterations
        self.cards = np.arange(decksize)
        self.bet_options = 2  # Check/fold (0) or Bet/call (1)
        self.nodes: Node = {} 

    def cfr_iterations_external(self):
        util = np.zeros(2)
        for t in range(1, self.iterations + 1):
            for i in range(2):
                #np.random.permutation(self.cards)
                random.shuffle(self.cards)
                util[i] += self.external_cfr(self.cards[:2], [],  2, 0, i, t)
        print('Average game value: {}'.format(util[0]   / (self.iterations)))

    def external_cfr(self, cards, history, pot, nodes_touched, traversing_player, t, reach_prob_0=1, reach_prob_1=1):
        plays = len(history)
        acting_player = plays % 2
        opponent_player = 1 - acting_player
        
        # Get reach probabilities
        my_reach = reach_prob_0 if acting_player == 0 else reach_prob_1
        opp_reach = reach_prob_1 if acting_player == 0 else reach_prob_0

        # Terminal states
        if plays >= 2:
            # Bet-fold sequence
            if history[-1] == 0 and history[-2] == 1:
                return 1 if acting_player == traversing_player else -1
            
            # Showdown after check-check or bet-call
            if (history[-1] == 0 and history[-2] == 0) or (history[-1] == 1 and history[-2] == 1):
                player_wins = cards[acting_player] > cards[opponent_player]
                if acting_player == traversing_player:
                    return pot/2 if player_wins else -pot/2
                else:
                    return -pot/2 if player_wins else pot/2

        # Get information set and create new node if needed
        # infoset = str(cards[acting_player]) + str(history)
        # For kuhn_evaluation
        infoset = str(cards[acting_player]) + ''.join(map(str, history))

        if infoset not in self.nodes:
            self.nodes[infoset] = Node(self.bet_options)

        nodes_touched += 1
        strategy = self.nodes[infoset].get_strategy(my_reach)  # Pass reach prob

        # Traversing player's turn - compute counterfactual values and update regrets
        if acting_player == traversing_player:
            util = np.zeros(self.bet_options)
            node_util = 0
            
            for action in range(self.bet_options):
                # Update reach probs based on player
                next_reach_0 = reach_prob_0 * strategy[action] if acting_player == 0 else reach_prob_0
                next_reach_1 = reach_prob_1 * strategy[action] if acting_player == 1 else reach_prob_1
                
                next_history = history + [action]
                next_pot = pot + action
                util[action] = self.external_cfr(cards, next_history, next_pot, nodes_touched, 
                                               traversing_player, t, next_reach_0, next_reach_1)
                node_util += strategy[action] * util[action]

            # Update regrets with opponent reach probability (Reach probability is the product of all action probabilities taken by a player to reach the current game state. )
            for action in range(self.bet_options):
                regret = util[action] - node_util
                self.nodes[infoset].regret_sum[action] += opp_reach * regret
            
            return node_util

        # Opponent's turn - sample action according to strategy
        else:
            action = np.random.choice(self.bet_options, p=strategy)
            next_history = history + [action]
            next_pot = pot + action
            
            # Accumulate strategy
            self.nodes[infoset].strategy_sum += strategy
            return self.external_cfr(cards, next_history, next_pot, nodes_touched, traversing_player, t)


if __name__ == "__main__":
    k = KuhnCFR(100000, 3)
    k.cfr_iterations_external()
    end_time = time.time()
    print("Elapsed time:", end_time - start_time, "seconds")
    print(k.nodes.keys())
    print(k.nodes.items())
