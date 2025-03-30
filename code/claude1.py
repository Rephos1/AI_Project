import numpy as np
import random
from collections import defaultdict

'''
Leduc Poker Rules:
- 2 betting rounds with a 2-bet maximum per round
- 2 suits × 3 ranks = 6 cards total
- Round 1: Each player antes 1 chip and is dealt 1 private card
- Round 1 betting: Check/Call, Bet/Raise (2 chips), or Fold
- Round 2: A single board card is revealed
- Round 2 betting: Check/Call, Bet/Raise (4 chips), or Fold
- Winning: Pair beats high card, higher pair wins, higher card wins
'''

# Actions: Check/Call, Bet/Raise, Fold
ACTIONS = ['c', 'b', 'f']
NUM_ACTIONS = len(ACTIONS)

# Card ranks
RANKS = 3  # Jack, Queen, King
SUITS = 2  # 2 suits
DECK_SIZE = RANKS * SUITS

# Round-specific constants
ANTE = 1
ROUND1_BET = 2
ROUND2_BET = 4


class Node:
    def __init__(self, info_set):
        self.info_set = info_set
        self.regret_sum = np.zeros(NUM_ACTIONS)
        self.strategy = np.zeros(NUM_ACTIONS)
        self.strategy_sum = np.zeros(NUM_ACTIONS)
        self.reach_prob_sum = 0
    
    def get_strategy(self, reach_prob):
        """
        Get current strategy through regret-matching
        """
        # Sum positive regrets
        normalizing_sum = 0
        for a in range(NUM_ACTIONS):
            self.strategy[a] = max(self.regret_sum[a], 0)
            normalizing_sum += self.strategy[a]
        
        # Normalize strategy
        if normalizing_sum > 0:
            for a in range(NUM_ACTIONS):
                self.strategy[a] /= normalizing_sum
        else:
            # Use uniform strategy if all regrets are negative or zero
            for a in range(NUM_ACTIONS):
                self.strategy[a] = 1.0 / NUM_ACTIONS
        
        # Accumulate strategy weighted by reach probability
        self.strategy_sum += reach_prob * self.strategy
        self.reach_prob_sum += reach_prob
        
        return self.strategy
    
    def get_average_strategy(self):
        """
        Get average strategy over all iterations
        """
        avg_strategy = np.zeros(NUM_ACTIONS)
        if self.reach_prob_sum > 0:
            for a in range(NUM_ACTIONS):
                avg_strategy[a] = self.strategy_sum[a] / self.reach_prob_sum
        else:
            # Default to uniform strategy if never reached
            for a in range(NUM_ACTIONS):
                avg_strategy[a] = 1.0 / NUM_ACTIONS
        
        return avg_strategy


class LeducCFR:
    def __init__(self, iterations=100000):
        self.iterations = iterations
        self.nodes = {}
        self.expected_game_value = 0
    
    def get_node(self, info_set):
        """
        Get or create a node for the given information set
        """
        if info_set not in self.nodes:
            self.nodes[info_set] = Node(info_set)
        return self.nodes[info_set]
    
    def is_terminal(self, history):
        """
        Check if the game has reached a terminal state
        """
        # If someone folded, the game is over
        if 'f' in history:
            return True
        
        # Count rounds and actions
        rounds = self.get_rounds(history)
        
        # If we've completed two rounds of betting, the game is over
        if len(rounds) > 1 and self.is_round_complete(rounds[1]):
            return True
            
        return False
    
    def get_rounds(self, history):
      """
      Split history into betting rounds
      """
      # Base case for empty history
      if not history:
          return []
      
      # If there's a fold, everything is in the first round
      if 'f' in history:
          return [history]
      
      # Simple approach: In Leduc, first round is complete after 2-4 actions
      # with balanced betting
      
      # Look for round boundary patterns
      patterns = ['cc', 'cbc', 'cbbc', 'bcbc', 'bcc', 'bbcc']
      
      for i in range(2, min(len(history) + 1, 7)):  # First round won't be longer than 6 actions
          prefix = history[:i]
          if any(prefix.endswith(p) for p in patterns):
              # Found first round boundary
              return [prefix, history[i:]] if i < len(history) else [prefix]
      
      # If no clear boundary found, assume all actions are in one round
      return [history] 
    
    def is_round_complete(self, round_history):
      """
      Check if a betting round is complete (both players have acted with equal bets)
      """
      # Convert round_history to a string if it's a list
      if isinstance(round_history, list):
          round_history = ''.join(round_history)
      
      if not round_history:
          return False
          
      # If someone folded, the round and game are over
      if 'f' in round_history:
          return True
          
      # Count actions per player
      actions_p0 = 0
      actions_p1 = 0
      bets_p0 = 0
      bets_p1 = 0
      
      for i, action in enumerate(round_history):
          player = i % 2
          if player == 0:
              actions_p0 += 1
              if action == 'b':
                  bets_p0 += 1
          else:
              actions_p1 += 1
              if action == 'b':
                  bets_p1 += 1
      
      # Both players have acted at least once
      if actions_p0 == 0 or actions_p1 == 0:
          return False
          
      # Check for common terminal patterns
      if round_history[-2:] == 'cc':  # Both checked
          return True
      if len(round_history) >= 2 and round_history[-1] == 'c' and round_history[-2] == 'b':  # Bet-call
          return True
      if len(round_history) >= 4 and round_history[-2:] == 'bc' and round_history[-4:-2] == 'bc':  # bet-call, bet-call
          return True
          
      return False  # Not complete otherwise
      
    def get_valid_actions(self, history):
        """
        Get valid actions for the current game state
        """
        # Fold is always available except when checking is free
        # Check/Call is always available
        # Bet/Raise is available if below the maximum number of bets for the round
        
        valid_actions = ['c']  # Always can check/call
        
        # Get current round
        rounds = self.get_rounds(history)
        current_round = rounds[-1] if rounds else ''
        
        # Determine the current player's position
        current_player = len(history) % 2
        
        # If the opponent has bet, we can fold
        if current_round and current_round[-1] == 'b':
            valid_actions.append('f')
        
        # Count bets in the current round
        bets_in_round = current_round.count('b')
        
        # Check if we can bet/raise (if under the 2-bet limit)
        if bets_in_round < 2:
            valid_actions.append('b')
            
        return valid_actions
    
    def get_round_number(self, history):
        """
        Determine which betting round we're in
        """
        rounds = self.get_rounds(history)
        return len(rounds)
    
    def evaluate_terminal(self, cards, history):
        """
        Calculate the utility in a terminal state
        """
        player_card = [cards[0], cards[1]]
        board_card = cards[2]
        
        # If someone folded, they lose the pot
        if 'f' in history:
            folding_player = -1
            for i, action in enumerate(history):
                if action == 'f':
                    folding_player = i % 2
                    break
                    
            winner = 1 - folding_player
            
            # Calculate pot size
            pot = self.calculate_pot_size(history)
            
            # Utility from winner's perspective
            util = np.zeros(2)
            util[winner] = pot
            util[1-winner] = -pot
            return util
        
        # Otherwise, compare hands
        # Convert card numbers to ranks (0-2)
        rank_p0 = player_card[0] % RANKS
        rank_p1 = player_card[1] % RANKS
        rank_board = board_card % RANKS
        
        winner = -1  # -1 means draw
        
        # Check for pairs
        if rank_p0 == rank_board and rank_p1 != rank_board:
            winner = 0
        elif rank_p1 == rank_board and rank_p0 != rank_board:
            winner = 1
        else:
            # No pairs - compare high cards
            if rank_p0 > rank_p1:
                winner = 0
            elif rank_p1 > rank_p0:
                winner = 1
        
        # Calculate pot size
        pot = self.calculate_pot_size(history)
        
        # In case of a draw (should be rare in Leduc)
        if winner == -1:
            return np.zeros(2)
        
        # Utility from winner's perspective
        util = np.zeros(2)
        util[winner] = pot
        util[1-winner] = -pot
        return util
    
    def calculate_pot_size(self, history):
        """
        Calculate the size of the pot based on betting history
        """
        pot = 2 * ANTE  # Both players ante
        
        rounds = self.get_rounds(history)
        
        # Add bets from first round
        if rounds:
            round1 = rounds[0]
            pot += round1.count('b') * ROUND1_BET
        
        # Add bets from second round
        if len(rounds) > 1:
            round2 = rounds[1]
            pot += round2.count('b') * ROUND2_BET
            
        return pot
    
    def cfr(self, reach_probs, cards, history=""):
        """
        Counterfactual regret minimization algorithm
        """
        player = len(history) % 2
        
        # Return expected value if we're at a terminal state
        if self.is_terminal(history):
            return self.evaluate_terminal(cards, history)
        
        # Get information set for current player
        round_num = self.get_round_number(history)
        
        # Info set includes player's card and public card (if in round 2)
        info_set = f"p{player}:c{cards[player]}"
        if round_num > 1:  # After first round
            info_set += f":b{cards[2]}"  # Board card
        info_set += f":{history}"
        
        # Get or create the current node
        node = self.get_node(info_set)
        
        # Get strategy for this information set
        strategy = node.get_strategy(reach_probs[player])
        
        # Initialize expected values
        action_utils = np.zeros(NUM_ACTIONS)
        
        # Get valid actions
        valid_actions = self.get_valid_actions(history)
        
        # Recursively compute utility for each action
        for a, action in enumerate(ACTIONS):
            if action in valid_actions:
                # Create new reach probabilities after taking action
                new_reach_probs = reach_probs.copy()
                new_reach_probs[player] *= strategy[a]
                
                # Recursively get utilities for this action
                action_utils[a] = self.cfr(new_reach_probs, cards, history + action)[player]
            else:
                # Invalid action, set utility to minimum
                action_utils[a] = float('-inf')
        
        # Compute expected value
        node_util = 0
        for a, action in enumerate(ACTIONS):
            if action in valid_actions:
                node_util += strategy[a] * action_utils[a]
        
        # Compute counterfactual regret
        for a, action in enumerate(ACTIONS):
            if action in valid_actions:
                regret = action_utils[a] - node_util
                
                # Update regret weighted by opponent's reach probability
                node.regret_sum[a] += reach_probs[1-player] * regret
        
        # Return expected utilities for both players
        return np.array([node_util, -node_util]) if player == 0 else np.array([-node_util, node_util])
    
    def train(self, iterations=None):
        """
        Train the CFR algorithm for the specified number of iterations
        """
        if iterations is not None:
            self.iterations = iterations
            
        util_sum = np.zeros(2)
        
        for i in range(self.iterations):
            # Create a shuffled deck
            deck = list(range(DECK_SIZE))
            random.shuffle(deck)
            
            # Deal cards (2 player cards + 1 board card)
            cards = deck[:3]
            
            # Start with uniform reach probabilities
            reach_probs = np.ones(2)
            
            # Run CFR and accumulate utilities
            util = self.cfr(reach_probs, cards)
            util_sum += util
            
            # Print progress occasionally
            if (i+1) % (self.iterations // 10) == 0:
                print(f"Iteration {i+1}/{self.iterations}: Average game value: {util_sum[0]/(i+1)}")
                
        # Save the expected game value
        self.expected_game_value = util_sum[0] / self.iterations
        
        return self.expected_game_value
    
    def get_strategy_profile(self):
        """
        Get the average strategy profile for all information sets
        """
        profile = {}
        
        for info_set, node in self.nodes.items():
            profile[info_set] = {
                'strategy': {ACTIONS[a]: prob for a, prob in enumerate(node.get_average_strategy())},
                'reach_prob': node.reach_prob_sum
            }
            
        return profile
    
    def print_strategy(self, min_reach_prob=0.001):
        """
        Print the computed strategy for each information set
        """
        profile = self.get_strategy_profile()
        
        # Sort by info sets
        sorted_infosets = sorted(profile.keys())
        
        print("\nStrategy Profile:")
        print("================")
        
        # for info_set in sorted_infosets:
        #     # Skip infrequently reached info sets
        #     if profile[info_set]['reach_prob'] < min_reach_prob:
        #         continue
                
        #     strategy = profile[info_set]['strategy']
            
        #     # Parse info set to get card info
        #     card_info, history = info_set.split(':', 1) if ':' in info_set else (info_set, "")
            
        #     print(f"Info set: {info_set}")
        #     print(f"  Reach probability: {profile[info_set]['reach_prob']:.6f}")
        #     print(f"  Strategy: Check/Call: {strategy['c']:.4f}, Bet/Raise: {strategy['b']:.4f}, Fold: {strategy['f']:.4f}")
        #     print()


# Example usage
if __name__ == "__main__":
    # Create and train the CFR algorithm
    leduc_cfr = LeducCFR(iterations=100000)  # Use more iterations for better convergence
    expected_value = leduc_cfr.train()
    
    print(f"\nExpected game value (player 0's perspective): {expected_value:.6f}")
    
    # Print the computed strategy profile
    leduc_cfr.print_strategy()