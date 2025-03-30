import numpy as np
import sys
sys.setrecursionlimit(10000)  # Increase recursion limit

'''
Leduc Rules:
2 rounds, 2-bet limit
2 suits * 3 cards per suit = 6 cards
1st round: 1 chip ante + single card dealt, chip bet = 2
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
        self.cards = np.arange(decksize)  # Cards 0-5: ranks 0 (0,1), 1 (2,3), 2 (4,5)
        self.node_map = {}

    def cfr_iterations_external(self):
        util = np.zeros(2)
        for _ in range(self.iterations):
            np.random.shuffle(self.cards)
            util += self.cfr_iterations_internal(self.cards, '', 1, 1)
        return util / self.iterations

    def cfr_iterations_internal(self, cards, history, p0, p1):
        # Add early return for deep recursion
        if len(history) > 20:  # Max reasonable game length
            return np.array([0, 0])
            
        plays = len(history)
        player = plays % 2

        # Terminal states check first
        if self.is_terminal(history):
            return self.evaluate_terminal(cards, history)

        # Get information set
        info_set = str(cards[player]) + history
        node = self.node_map.get(info_set)
        if node is None:
            node = Node()
            node.infoset = info_set
            self.node_map[info_set] = node

        # Get strategy and initialize utilities
        strategy = node.get_strategy(p0 if player == 0 else p1)
        util = np.zeros(NUM_ACTIONS)
        node_util = 0

        # Recursive case with better handling
        for a, action in enumerate(ACTIONS):
            next_history = history + action
            
            # Handle end of first round
            if len(next_history) >= 2 and next_history[-2:] in ['cc', 'bc'] and '|' not in history and action != 'f':
                next_history += '|' + str(cards[2])

            # Recursive call with proper reach probabilities
            if player == 0:
                util[a] = -self.cfr_iterations_internal(
                    cards, 
                    next_history,
                    p0 * strategy[a],  # Update reach prob for player 0
                    p1
                )[1]  # Get opponent's utility
            else:
                util[a] = -self.cfr_iterations_internal(
                    cards,
                    next_history,
                    p0,
                    p1 * strategy[a]  # Update reach prob for player 1
                )[0]  # Get opponent's utility
                
            node_util += strategy[a] * util[a]

        # Update regrets
        for a in range(NUM_ACTIONS):
            regret = util[a] - node_util
            if player == 0:
                node.regret_sum[a] += p1 * regret  # Weight by opponent reach prob
            else:
                node.regret_sum[a] += p0 * regret

        return np.array([-node_util, node_util]) if player == 1 else np.array([node_util, -node_util])

    def is_terminal(self, history):
        if history.endswith('f'):
            return True
        if '|' in history:
            second_round_history = history.split('|')[1]
            return len(second_round_history) >= 2 and second_round_history[-2:] in ['cc', 'bc']
        return False

    def evaluate_terminal(self, cards, history):
        if history.endswith('f'):
            winner = len(history) % 2  # Last player folded, other wins
        else:
            # Extract board card
            board_card = int(history.split('|')[1].strip()[0])
            p0_has_pair = (cards[0] // 2 == board_card // 2)
            p1_has_pair = (cards[1] // 2 == board_card // 2)
            if p0_has_pair and not p1_has_pair:
                winner = 0
            elif p1_has_pair and not p0_has_pair:
                winner = 1
            elif p0_has_pair and p1_has_pair:
                winner = -1  # Tie
            else:
                p0_rank = cards[0] // 2
                p1_rank = cards[1] // 2
                winner = 0 if p0_rank > p1_rank else (1 if p1_rank > p0_rank else -1)

        # Approximate pot; ideally, compute exactly based on rounds
        pot = 2 + 2 * history.count('b')  # Still approximate
        if winner == -1:
            return np.array([0, 0])  # Split pot in tie
        return np.array([pot if winner == 0 else -pot, -pot if winner == 0 else pot])

# Example usage
cfr = LeducCFR(iterations=3900000)
average_utility = cfr.cfr_iterations_external()
print(f"Average utility after CFR iterations: {average_utility}")