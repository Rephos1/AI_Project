import numpy as np

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
            iter_util = self.cfr_iterations_internal(self.cards[:3], '', 1, 1)
            util += iter_util
            if i % 1000 == 0:
                print(f"Iteration {i}, utility: {iter_util}")
        return util / self.iterations

    def cfr_iterations_internal(self, cards, history, p0, p1):
        player = len(history) % 2
        
        if self.is_terminal(history):
            return self.evaluate_terminal(cards, history)
        
        valid_actions = self.get_valid_actions(history)
        
        round1_end = self.is_round1_over(history)
        
        if round1_end > 0 and len(history) >= round1_end:
            info_set = f"{cards[player]}b{cards[2]}{history[round1_end:]}"
        else:
            info_set = f"{cards[player]}{history}"
        
        if info_set not in self.node_map:
            self.node_map[info_set] = Node()
            self.node_map[info_set].infoset = info_set
        
        node = self.node_map[info_set]
        
        strategy = node.get_strategy(p0 if player == 0 else p1)
        
        util = np.zeros(NUM_ACTIONS)
        node_util = 0
        
        for a, action in enumerate(ACTIONS):
            if action not in valid_actions:
                continue
                
            next_history = history + action
            
            if player == 0:
                action_util = self.cfr_iterations_internal(cards, next_history, p0 * strategy[a], p1)
                util[a] = action_util[0]
            else:
                action_util = self.cfr_iterations_internal(cards, next_history, p0, p1 * strategy[a])
                util[a] = action_util[1]
                
            node_util += strategy[a] * util[a]
        
        for a, action in enumerate(ACTIONS):
            if action in valid_actions:
                regret = util[a] - node_util
                counterfactual_prob = p1 if player == 0 else p0
                node.regret_sum[a] += counterfactual_prob * regret
        
        result = np.zeros(2)
        result[player] = node_util
        result[1-player] = -node_util
        return result
        
    def get_valid_actions(self, history):
        if 'f' in history:
            return []
            
        last_action = history[-1] if history else None
        
        if last_action == 'b':
            round1_end = self.is_round1_over(history)
            if round1_end == 0:
                current_round_history = history
                bet_count = current_round_history.count('b')
                if bet_count >= 2:
                    return ['c', 'f']
                else:
                    return ['c', 'b', 'f']
            else:
                round2_history = history[round1_end:]
                bet_count = round2_history.count('b')
                if bet_count >= 2:
                    return ['c', 'f']
                else:
                    return ['c', 'b', 'f']
        elif last_action == 'c':
            return ['c', 'b']
        else:
            return ['c', 'b']
            
    def is_round1_over(self, history):
        if 'f' in history:
            return 0
        
        for i in range(1, len(history) + 1):
            substr = history[:i]
            if len(substr) >= 2 and substr.endswith('cc'):
                return i
            if len(substr) >= 2 and substr.endswith('bc'):
                return i
            if len(substr) >= 3 and substr.endswith('bbc'):
                return i
        return 0

    def is_terminal(self, history):
        if 'f' in history:
            return True
            
        round1_end = self.is_round1_over(history)
        if round1_end == 0:
            return False
            
        round2_history = history[round1_end:]
        if not round2_history:
            return False
        
        for i in range(1, len(round2_history) + 1):
            substr = round2_history[:i]
            if substr.endswith('cc') or substr.endswith('bc') or substr.endswith('bbc'):
                return True
        return False

    def evaluate_terminal(self, cards, history):
        pot = 2  
        
        if 'f' in history:
            folding_player = history.index('f') % 2
            winner = 1 - folding_player
            
            for i, action in enumerate(history[:history.index('f')]):
                round_num = 1 if i < self.is_round1_over(history[:i]) else 2
                if action == 'b':
                    pot += 2 if round_num == 1 else 4
                elif action == 'c' and i > 0 and history[i-1] == 'b':
                    pot += 2 if round_num == 1 else 4
        else:
            player0_rank = cards[0] % 3
            player1_rank = cards[1] % 3
            board_rank = cards[2] % 3
            
            if player0_rank == board_rank or player1_rank == board_rank:
                if player0_rank == board_rank and player1_rank != board_rank:
                    winner = 0
                elif player1_rank == board_rank and player0_rank != board_rank:
                    winner = 1
                else:
                    winner = 0 if player0_rank > player1_rank else 1
            else:
                winner = 0 if player0_rank > player1_rank else 1
            
            round1_end = self.is_round1_over(history)
            
            for i, action in enumerate(history[:round1_end]):
                if action == 'b':
                    pot += 2
                elif action == 'c' and i > 0 and history[i-1] == 'b':
                    pot += 2
                    
            for i, action in enumerate(history[round1_end:]):
                if action == 'b':
                    pot += 4
                elif action == 'c' and i > 0 and history[round1_end + i -1] == 'b':
                    pot += 4
        
        return np.array([pot if winner == 0 else -pot, -pot if winner == 0 else pot])

    def print_strategies(self):
        print("\nComputed strategies:")
        for info_set, node in sorted(self.node_map.items()):
            avg_strategy = node.get_average_strategy()
            if sum(avg_strategy) > 0.99:
                print(f"Info set: {info_set}, Strategy: {dict(zip(ACTIONS, avg_strategy))}")


if __name__ == "__main__":
    cfr = LeducCFR(iterations=100000)
    average_utility = cfr.cfr_iterations_external()
    print(f"Average game value: {average_utility}")
    cfr.print_strategies()