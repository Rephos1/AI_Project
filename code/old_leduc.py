import numpy as np

'''
Leduc Rules:
2 rounds, 2-bet limit
2 suits * 3 cards per = 6 cards
1st round: 1 chip ante + single card is dealt, chip bet = 2
2nd round: 1 board card revealed, another betting round, chip bet = 4 
'''

ACTIONS = ['c', 'b', 'f'] # Check/Call, Bet/Raise, Fold
NUM_ACTIONS = len(ACTIONS)
TERMINAL_HISTORIES_AGNOSTIC = ['f','cc','cbf','cbc','cbbf','cbbc','bf','bc','bbf','bbc'] # Terminal histories for Leduc Poker

class Node:
    def __init__(self):
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
        for _ in range(self.iterations):
            np.random.shuffle(self.cards)
            util += self.cfr_iterations_internal(self.cards, '', 1, 1)
        return util / self.iterations

    def cfr_iterations_internal(self, cards, history, p0, p1):
        plays = len(history)
        player = plays % 2 if 'r' not in history else (plays - 1) % 2
        round1 = self.is_r1(history)
        if not round1 and history in TERMINAL_HISTORIES_AGNOSTIC and len(history) <= 4:
            history = history + 'r'
        if round1 and history in TERMINAL_HISTORIES_AGNOSTIC and 'r' not in history and len(history) <= 4:
            history = history + 'r'
        if self.is_terminal(history):
            return self.evaluate_terminal(cards, history)

        info_set = str(cards[player]) + history
        if info_set not in self.node_map:
            node = Node()
            node.infoset = info_set
            self.node_map[info_set] = node
        else:
            node = self.node_map[info_set]

        strategy = node.get_strategy(p0 if player == 0 else p1)
        util = np.zeros(NUM_ACTIONS)
        node_util = 0

        for a, action in enumerate(ACTIONS):
            next_history = history + action
            if player == 0:
                util[a] = -self.cfr_iterations_internal(cards, next_history, p0 * strategy[a], p1)[player]
            else:
                util[a] = -self.cfr_iterations_internal(cards, next_history, p0, p1 * strategy[a])[player]
            node_util += strategy[a] * util[a]

        for a in range(NUM_ACTIONS):
            regret = util[a] - node_util
            node.regret_sum[a] += (p1 if player == 0 else p0) * regret

        return np.array([node_util, -node_util])

    def is_r1(self, history):
        if 'r' in history:
            return True
        elif history in TERMINAL_HISTORIES_AGNOSTIC and 'r' not in history:
            return True
        return False
    
    def is_terminal(self, history):
        round1 = self.is_r1(history)
        
        if round1:
            print(history)
            r1_length = history.index('r')
        if round1 and history[r1_length+1:] in TERMINAL_HISTORIES_AGNOSTIC:
            return True
        if history.endswith('f'):
            return True
        return len(history) >= 2 and history in TERMINAL_HISTORIES_AGNOSTIC

    def evaluate_terminal(self, cards, history):
        player_folded = history.endswith('f')
        if player_folded:
            winner = (len(history) + 1) % 2
        else:
            winner = 0 if cards[0] // 2 > cards[1] // 2 else 1
        pot = 2 + 2 * history.count('b')
        return np.array([pot if winner == 0 else -pot, -pot if winner == 0 else pot])


# Example usage
cfr = LeducCFR(iterations=300000)
average_utility = cfr.cfr_iterations_external()
print(f"Average utility after CFR iterations: {average_utility}")