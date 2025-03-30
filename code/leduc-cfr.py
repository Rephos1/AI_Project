import numpy as np

'''
Leduc Rules:
2 rounds, 2-bet limit
2 suits * 3 cards per = 6 cards
1st round: 1 chip ante + single card is dealt, chip bet = 2
2nd round: 1 board card revealed, another betting round, chip bet = 4 
'''

ACTIONS = ['c', 'b', 'f']  # Check/Call, Bet/Raise, Fold
NUM_ACTIONS = len(ACTIONS)

class Node:
    def __init__(self):
        self.infoset = ''
        self.num_actions = NUM_ACTIONS
        self.regret_sum = np.zeros(self.num_actions)
        self.strategy = np.zeros(self.num_actions)
        self.strategy_sum = np.zeros(self.num_actions)

    def get_strategy(self, reaching_prob: float):
        normalizing_sum = 0
        for a in range(self.num_actions):
            self.strategy[a] = max(self.regret_sum[a], 0)
            normalizing_sum += self.strategy[a]

        for a in range(self.num_actions):
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 1.0 / self.num_actions
            self.strategy_sum[a] += reaching_prob * self.strategy[a]

        return self.strategy

    def get_average_strategy(self):
        normalizing_sum = sum(self.strategy_sum)
        avg_strategy = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            avg_strategy[a] = (
                self.strategy_sum[a] / normalizing_sum if normalizing_sum > 0 else 1.0 / self.num_actions
            )
        return avg_strategy


class LeducCFR:
    def __init__(self, iterations, decksize=6):
        self.iterations = iterations
        self.decksize = decksize
        self.cards = np.arange(decksize)
        self.node_map = {}

    def cfr_iterations_external(self):
        util = np.zeros(2)
        for i in range(self.iterations):
            np.random.shuffle(self.cards)
            # First 2 cards for players, third card for the board
            iter_util = self.cfr_iterations_internal(self.cards[:3], '', 1, 1)
            util += iter_util
            if i % 1000 == 0:
                print(f"Iteration {i}, utility: {iter_util}")
        return util / self.iterations

    def cfr_iterations_internal(self, cards, history, p0, p1):
        player = len(history) % 2
        
        # Check for terminal states first
        if self.is_terminal(history):
            return self.evaluate_terminal(cards, history)
        
        # Get valid actions
        valid_actions = self.get_valid_actions(history)
        
        # Determine if we're in round 2
        round1_end = self.is_round1_over(history)
        
        # Construct the info set
        if round1_end > 0 and len(history) >= round1_end:
            # In round 2, include board card
            info_set = f"{cards[player]}b{cards[2]}{history[round1_end:]}"
        else:
            # In round 1, just player's card
            info_set = f"{cards[player]}{history}"
        
        # Get or create the node for this information set
        if info_set not in self.node_map:
            self.node_map[info_set] = Node()
            self.node_map[info_set].infoset = info_set
        
        node = self.node_map[info_set]
        
        # Get current strategy
        strategy = node.get_strategy(p0 if player == 0 else p1)
        
        # Calculate expected utility for each action
        util = np.zeros(NUM_ACTIONS)
        node_util = 0
        
        # Recurse for each valid action
        for a, action in enumerate(ACTIONS):
            if action not in valid_actions:
                continue
                
            next_history = history + action
            
            # Calculate utility recursively
            if player == 0:
                action_util = self.cfr_iterations_internal(cards, next_history, p0 * strategy[a], p1)
                util[a] = action_util[0]  # Player 0's perspective
            else:
                action_util = self.cfr_iterations_internal(cards, next_history, p0, p1 * strategy[a])
                util[a] = action_util[1]  # Player 1's perspective
                
            node_util += strategy[a] * util[a]
        
        # Update regrets
        for a, action in enumerate(ACTIONS):
            if action in valid_actions:
                regret = util[a] - node_util
                counterfactual_prob = p1 if player == 0 else p0
                node.regret_sum[a] += counterfactual_prob * regret
        
        # Return utilities from the current player's perspective
        result = np.zeros(2)
        result[player] = node_util
        result[1-player] = -node_util
        return result
        
    def get_valid_actions(self, history):
        """Returns list of valid actions in current state"""
        # At the start of the game
        if not history:
            return ['c', 'b']  # Can check or bet
            
        # If someone folded, no more actions
        if 'f' in history:
            return []
            
        # Last action matters
        last_action = history[-1]
        
        # After a bet
        if last_action == 'b':
            # Can call, raise, or fold after a bet
            # Need to check bet limits per round
            round1_end = self.is_round1_over(history)
            
            if round1_end == 0:  # In round 1
                bet_count = history.count('b')
                if bet_count >= 2:  # Already 2 bets in round 1
                    return ['c', 'f']  # Can only call or fold
                else:
                    return ['c', 'b', 'f']  # Can call, raise, or fold
            else:  # In round 2
                round2_history = history[round1_end:]
                bet_count = round2_history.count('b')
                if bet_count >= 2:  # Already 2 bets in round 2
                    return ['c', 'f']  # Can only call or fold
                else:
                    return ['c', 'b', 'f']  # Can call, raise, or fold
        
        # After a check/call
        elif last_action == 'c':
            # Can check or bet, but not fold (nothing to fold to)
            return ['c', 'b']
            
        # Something went wrong - should never get here in valid game
        return []
            
    def is_round1_over(self, history):
        """Determine if Round 1 is over and return the index where Round 2 starts"""
        if not history:
            return 0
        if 'f' in history:
            return 0  # Game is over, not Round 2
        for i in range(2, len(history) + 1):
            prefix = history[:i]
            if prefix.endswith('cc') or prefix.endswith('bc') or prefix.endswith('bbc'):
                return i
        return 0  # Still in Round 1

    def is_terminal(self, history):
        """Determine if the game has reached a terminal state"""
        # Game ends if someone folds
        if 'f' in history:
            return True
            
        # Check if we're in round 2
        round1_end = self.is_round1_over(history)
        if round1_end == 0:  # Still in round 1
            return False
            
        # If we just finished round 1, not terminal yet
        if round1_end == len(history):
            return False
            
        # Get round 2 history
        round2_history = history[round1_end:]
        
        # Terminal cases for round 2
        if round2_history.endswith('cc'):  # Both check
            return True
        elif round2_history.endswith('bc'):  # Bet and call
            return True
        elif round2_history.endswith('bbc'):  # Bet, raise, call
            return True
            
        return False

    def evaluate_terminal(self, cards, history):
        """Determine the payoff at a terminal state"""
        # Initialize pot with 1 ante chip from each player
        pot = 2  
        
        # Process bets
        if 'f' in history:
            # Someone folded - other player wins
            folding_player = history.index('f') % 2
            winner = 1 - folding_player
            
            # Count bets before fold
            for i, action in enumerate(history[:history.index('f')]):
                round_num = 1
                if i >= self.is_round1_over(history[:i]) > 0:
                    round_num = 2
                    
                if action == 'b':
                    pot += 2 if round_num == 1 else 4
                elif action == 'c' and i > 0 and history[i-1] == 'b':
                    pot += 2 if round_num == 1 else 4
        else:
            # Game went to showdown - compare hands
            player0_rank = cards[0] % 3  # Card rank (0, 1, or 2)
            player1_rank = cards[1] % 3  # Card rank
            board_rank = cards[2] % 3    # Board card rank
            
            # Check for pairs with the board card
            if player0_rank == board_rank and player1_rank != board_rank:
                winner = 0  # Player 0 has a pair
            elif player1_rank == board_rank and player0_rank != board_rank:
                winner = 1  # Player 1 has a pair
            elif player0_rank == board_rank and player1_rank == board_rank:
                # Both have pairs, compare ranks (though this is impossible in standard Leduc)
                winner = 0 if player0_rank > player1_rank else 1
            else:
                # No pairs, high card wins
                winner = 0 if player0_rank > player1_rank else 1
                
            # Count all bets for showdown
            round1_end = self.is_round1_over(history)
            
            # Round 1 bets
            for i, action in enumerate(history[:round1_end]):
                if action == 'b':
                    pot += 2  # Bet in round 1
                elif action == 'c' and i > 0 and history[i-1] == 'b':
                    pot += 2  # Call in round 1
                    
            # Round 2 bets
            for i, action in enumerate(history[round1_end:]):
                if action == 'b':
                    pot += 4  # Bet in round 2
                elif action == 'c' and i > 0 and history[round1_end+i-1] == 'b':
                    pot += 4  # Call in round 2
        
        # Return payoff from each player's perspective
        return np.array([pot if winner == 0 else -pot, -pot if winner == 0 else pot])

    def print_strategies(self):
        """Print the computed strategies"""
        print("\nComputed strategies:")
        for info_set, node in sorted(self.node_map.items()):
            avg_strategy = node.get_average_strategy()
            if sum(avg_strategy) > 0.99:  # Only print meaningful strategies
                print(f"Info set: {info_set}, Strategy: {dict(zip(ACTIONS, avg_strategy))}")


# Example usage
if __name__ == "__main__":
    cfr = LeducCFR(iterations=100000)
    average_utility = cfr.cfr_iterations_external()
    print(f"Average game value: {average_utility}")
    cfr.print_strategies()