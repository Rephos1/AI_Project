import numpy as np
import random
import time
import sys
# Increase recursion limit

MAX_DEPTH = 100  # Maximum depth of game tree to prevent infinite recursion

'''
Leduc Rules:
2 rounds, 2-bet limit
2 suits * 3 cards per = 6 cards
1st round: 1 chip ante +  single card is dealt, chip bet = 2
2nd round: 1 board card revealed, another betting round, chip bet is 4 

'''
ACTIONS = 3 # Check/Fold  Bet/Call Reraise

class Node:
    def __init__(self):
        self.infoset = ''
        self.num_actions = ACTIONS
        self.regret_sum = np.zeros(self.num_actions)
        self.strategy = np.zeros(self.num_actions)
        self.strategy_sum = np.zeros(self.num_actions)
    
    def get_strategy(self, reaching_prob: float):
        # Calculate strategy from regrets
        positive_regrets = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(positive_regrets)
        
        # Calculate current strategy
        if normalizing_sum > 0:
            self.strategy = positive_regrets / normalizing_sum
        else:
            # Use uniform strategy if no positive regrets
            self.strategy = np.ones(self.num_actions) / self.num_actions
        
        # Accumulate strategy weighted by reach probability
        self.strategy_sum += reaching_prob * self.strategy
        return self.strategy


class LeducCFR:
    def __init__(self, iterations, deck_size=6):
        self.iterations = iterations
        self.deck_size = deck_size
        self.cards = np.arange(deck_size)
        self.nodes = {}
        self.ante = 1
        self.bet_options = ACTIONS  
        self.raise_size = {
            1: 2,  # First round raise size
            2: 4   # Second round raise size
        }

    def terminal_state_value(self, history, cards, pot, acting_player, traversing_player):
        """Returns the utility value at terminal states"""
        opponent = 1 - acting_player
        
        # Round 1 fold
        if 'r' not in history and history[-1] == 'f':
            return pot if acting_player == traversing_player else -pot
            
        # Round 2 fold
        if 'r' in history and history[-1] == 'f':
            return pot if acting_player == traversing_player else -pot
        
        # Showdown
        player_card = cards[acting_player]
        opponent_card = cards[opponent]
        community_card = cards[2]  # Third card is the community card
        
        # Pair evaluation
        player_pair = player_card == community_card
        opponent_pair = opponent_card == community_card
        
        # Determine winner
        if player_pair and not opponent_pair:
            winner = acting_player
        elif opponent_pair and not player_pair:
            winner = opponent
        else:  # No pairs or both pairs - compare high cards
            winner = acting_player if player_card > opponent_card else opponent
            
        if winner == traversing_player:
            return pot/2
        else:
            return -pot/2

    def is_terminal(self, history):
        """Checks if the game state is terminal"""
        # Fold in any round
        if history and history[-1] == 'f':
            return True
            
        # Round 1 checks
        if 'r' not in history:
            if len(history) >= 2 and history[-2:] in ['cc', 'bc', 'rc']:
                return False  # Move to round 2
        
        # Round 2 checks
        else:
            if history[-2:] in ['cc', 'bc', 'rc']:
                return True
        
        return False

    def valid_actions(self, history):
        """Returns list of valid actions at current state"""
        if not history:  # Start of round 1
            return ['c', 'r']  # Check or Raise
            
        last_action = history[-1]
        
        if 'r' not in history:  # Round 1
            if last_action == 'c':
                return ['c', 'r']  # Check-Check or Check-Raise
            if last_action == 'r':
                return ['f', 'c', 'r']  # Fold, Call, or Re-raise
        else:  # Round 2
            if last_action == 'c':
                return ['c', 'r']
            if last_action == 'r':
                return ['f', 'c', 'r']
                
        return ['f', 'c']  # Default options

    def calculate_next_pot(self, pot, action, history):
        """Calculates the next pot size based on the action taken"""
        if action == 'c':  # Check
            return pot
        elif action == 'f':  # Fold
            return pot
        elif action == 'r':  # Raise
            round_number = 1 if 'r' not in history else 2
            return pot + self.raise_size[round_number]
        return pot

    def cfr_iterations_external(self):
        util = np.zeros(2)
        for t in range(1, self.iterations + 1):
            # Deal cards: 2 player cards + 1 community card
            random.shuffle(self.cards)
            cards = self.cards[:3]  # First 2 for players, 3rd is community
            for i in range(2):
                util[i] += self.external_cfr(cards, [], self.ante * 2, 0, i, t)
        print('Average game value: {}'.format(util[0] / self.iterations))
    
    def external_cfr(self, cards, history, pot, nodes_touched, traversing_player, t, reach_prob_0=1, reach_prob_1=1):
        # Add depth check
        if len(history) > MAX_DEPTH:
            return 0

        plays = len(history)
        acting_player = plays % 2
        opponent_player = 1 - acting_player
        
        # Get reach probabilities
        my_reach = reach_prob_0 if acting_player == 0 else reach_prob_1
        opp_reach = reach_prob_1 if acting_player == 0 else reach_prob_0

        # Terminal states
        if self.is_terminal(history):
            return self.terminal_state_value(history, cards, pot, acting_player, traversing_player)

        # Get information set and create new node if needed
        infoset = str(cards[acting_player]) + str(history)
        if infoset not in self.nodes:
            self.nodes[infoset] = Node()

        nodes_touched += 1
        
        # Get valid actions
        valid_acts = self.valid_actions(history)
        num_actions = len(valid_acts)
        
        # Get strategy
        strategy = self.nodes[infoset].get_strategy(my_reach)
        
        if acting_player == traversing_player:
            util = np.zeros(num_actions)
            node_util = 0
            
            for i, action in enumerate(valid_acts):
                next_reach_0 = reach_prob_0 * strategy[i] if acting_player == 0 else reach_prob_0
                next_reach_1 = reach_prob_1 * strategy[i] if acting_player == 1 else reach_prob_1
                
                next_history = history + [action]
                next_pot = self.calculate_next_pot(pot, action, history)
                util[i] = self.external_cfr(cards, next_history, next_pot, nodes_touched, 
                                          traversing_player, t, next_reach_0, next_reach_1)
                node_util += strategy[i] * util[i]

            # Update regrets
            for i, action in enumerate(valid_acts):
                regret = util[i] - node_util
                self.nodes[infoset].regret_sum[i] += opp_reach * regret
            
            return node_util

        else:
            # Sample action according to strategy
            action_probs = strategy[:num_actions]
            # Ensure probabilities sum to 1
            action_probs = action_probs / np.sum(action_probs)
            action_idx = np.random.choice(num_actions, p=action_probs)
            action = valid_acts[action_idx]
            next_history = history + [action]
            next_pot = self.calculate_next_pot(pot, action, history)
            
            return self.external_cfr(cards, next_history, next_pot, nodes_touched, traversing_player, t)


if __name__ == "__main__":
    k = LeducCFR(300000, 3)
    start_time = time.time()
    k.cfr_iterations_external()
    end_time = time.time()
    print("Elapsed time:", end_time - start_time, "seconds")
    print(k.nodes.keys())
    print(k.nodes.items())




    '''Checks for leduc:
       1.if h ends with 'f' 
       2.have 'b','c','f' as actions
       3. separate rounds with 'r'
       4. if r2 and h[-1] =='f' or h == 'cbb' or h == 'cbc' or h == 'cc' or 'h == bc' or h == 'bbc'
       '''

