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
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / self.num_actions
        return avg_strategy


class LeducCFR:
    def __init__(self, iterations, decksize=6):
        self.iterations = iterations
        self.decksize = decksize
        self.cards = np.arange(decksize)
        self.node_map = {}

    def cfr_iterations_external(self):
        util = np.zeros(2)
        for _ in range(self.iterations):
            np.random.shuffle(self.cards)
            # First 2 cards for players, third is the board card
            util += self.cfr_iterations_internal(self.cards[:3], '', 1, 1)
        return util / self.iterations

    def cfr_iterations_internal(self, cards, history, p0, p1):
        plays = len(history)
        current_player = plays % 2
        
        # Check if we're in a terminal state
        if self.is_terminal(history):
            return self.evaluate_terminal(cards, history)

        # Determine which round we're in
        round_num = 0
        if plays >= 2:
            # Count number of betting rounds completed
            round_actions = 0
            for i in range(len(history)):
                if history[i] in 'cb':
                    round_actions += 1
                # A fold ends everything
                if history[i] == 'f':
                    break
                # If we see a complete round (both players acted, no fold)
                if round_actions == 2 and i == 1:
                    round_num = 1
        
        # Build information set string - player card + public card (if revealed) + history
        info_set = str(cards[current_player])
        if round_num == 1:  # Second round, board card is visible
            info_set += ":" + str(cards[2])
        info_set += ":" + history
        
        # Get or create the node for this information set
        if info_set not in self.node_map:
            node = Node()
            node.infoset = info_set
            self.node_map[info_set] = node
        else:
            node = self.node_map[info_set]

        # Get current strategy
        strategy = node.get_strategy(p0 if current_player == 0 else p1)
        util = np.zeros(NUM_ACTIONS)
        node_util = 0

        # Recursively compute utility for each action
        for a, action in enumerate(ACTIONS):
            if self.is_valid_action(history, action):
                next_history = history + action
                if current_player == 0:
                    util[a] = -self.cfr_iterations_internal(cards, next_history, p0 * strategy[a], p1)[current_player]
                else:
                    util[a] = -self.cfr_iterations_internal(cards, next_history, p0, p1 * strategy[a])[current_player]
                node_util += strategy[a] * util[a]
            else:
                # Invalid action (e.g., can't fold when checking is free)
                util[a] = -1000  # Heavy penalty

        # Update regrets
        for a in range(NUM_ACTIONS):
            if self.is_valid_action(history, action):
                regret = util[a] - node_util
                node.regret_sum[a] += (p1 if current_player == 0 else p0) * regret

        return np.array([node_util, -node_util])

    def is_valid_action(self, history, action):
        # Basic validation - in a real implementation this would be more detailed
        if action == 'f' and (len(history) == 0 or history[-1] == 'c'):
            return False  # Can't fold when checking is free
        return True
    
    def is_terminal(self, history):
        # A fold always ends the game
        if 'f' in history:
            return True
            
        # Count actions in the history
        if len(history) < 2:
            return False  # Need at least 2 actions to end a round
            
        # Check if we've completed two rounds
        round1_complete = False
        actions_count = 0
        
        for i, action in enumerate(history):
            if action in 'cb':
                actions_count += 1
            
            # First round ends after both players have acted
            if actions_count == 2 and not round1_complete:
                round1_complete = True
                actions_count = 0
            
            # Second round ends after both players have acted
            elif actions_count == 2 and round1_complete:
                return True
                
        # Additional checks for round completion with raises
        # This is simplified - a full implementation would track bets more carefully
        if 'bb' in history[-2:] or ('bc' in history[-2:] and round1_complete):
            return True
            
        return False

    def evaluate_terminal(self, cards, history):
        # Check if someone folded
        if 'f' in history:
            folding_player = history.rfind('f')
            # The player who didn't fold wins the pot
            winner = (folding_player + 1) % 2
            pot = 2  # Antes
            # Count bets
            for action in history:
                if action == 'b':
                    pot += 2  # First round bet
                    if history.index(action) > 2:  # Assuming second round starts after 2 actions
                        pot += 2  # Additional for second round bet
            return np.array([pot if winner == 0 else -pot, -pot if winner == 0 else pot])
        
        # Otherwise, compare hands
        player0_card = cards[0] % 3  # Get card rank (0, 1, or 2)
        player1_card = cards[1] % 3
        board_card = cards[2] % 3
        
        # Check for pairs
        if player0_card == board_card and player1_card != board_card:
            winner = 0
        elif player1_card == board_card and player0_card != board_card:
            winner = 1
        # High card wins
        elif player0_card > player1_card:
            winner = 0
        elif player1_card > player0_card:
            winner = 1
        else:
            # Tie (this shouldn't happen in standard Leduc)
            return np.zeros(2)
            
        # Calculate pot
        pot = 2  # Antes
        first_round_bets = 0
        second_round_bets = 0
        
        round1_complete = False
        actions_count = 0
        
        for action in history:
            if action == 'b':
                if not round1_complete:
                    first_round_bets += 2
                else:
                    second_round_bets += 4
            
            if action in 'cb':
                actions_count += 1
            
            if actions_count == 2 and not round1_complete:
                round1_complete = True
                actions_count = 0
                
        pot += first_round_bets + second_round_bets
            
        return np.array([pot if winner == 0 else -pot, -pot if winner == 0 else pot])


# Example usage
cfr = LeducCFR(iterations=200000)  # Reduced for example
average_utility = cfr.cfr_iterations_external()
print(f"Average utility after CFR iterations: {average_utility}")