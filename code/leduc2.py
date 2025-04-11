import numpy as np
import random
import time

# --- Global definitions ---
# Allowed actions (for both rounds)
ACTIONS = ['c', 'b', 'f']  # c: check/call, b: bet/raise, f: fold
NUM_ACTIONS = len(ACTIONS)

# --- Node class ---
class Node:
    def __init__(self, infoset, num_actions=NUM_ACTIONS):
        self.infoset = infoset
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)

    def get_strategy(self, reach_prob=1.0):
        normalizing_sum = 0.0
        for a in range(self.num_actions):
            # Use only positive regrets
            self.strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0.0
            normalizing_sum += self.strategy[a]
        for a in range(self.num_actions):
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 1.0 / self.num_actions
            self.strategy_sum[a] += reach_prob * self.strategy[a]
        return self.strategy

    def get_average_strategy(self):
        normalizing_sum = np.sum(self.strategy_sum)
        avg_strategy = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / self.num_actions
        return avg_strategy

# --- Leduc Poker CFR Implementation ---
class LeducCFR:
    def __init__(self, iterations):
        self.iterations = iterations
        self.deck = list(range(6))   # Leduc deck: 2 suits * 3 ranks = 6 cards (cards 0..5)
        self.node_map = {}           # dictionary for storing nodes

        # Game parameters (as given in the rules):
        self.ante = 1              # 1 chip ante per player
        self.bet_r1 = 2            # Bet size in round 1 is 2 chips
        self.bet_r2 = 4            # Bet size in round 2 is 4 chips

    def cfr(self):
        util = np.zeros(2)  # cumulative utility for player 0 and 1
        for t in range(1, self.iterations + 1):
            random.shuffle(self.deck)
            # Deal private cards:
            p0_card = self.deck[0]
            p1_card = self.deck[1]
            # Initial pot is 2 (antes from both players)
            pot = 2 * self.ante
            history_r1 = ""  # Start with empty history for round 1

            # Run external CFR for round 1 for each traversing player.
            # (We update CFR regret for each player as traversing.)
            for traversing in [0, 1]:
                util[traversing] += self.external_cfr_r1(p0_card, p1_card, history_r1, pot, traversing, 1.0, 1.0)
        return util / self.iterations

    def external_cfr_r1(self, p0_card, p1_card, history, pot, traversing, rp0, rp1):
        """
        External CFR recursion for round 1 (no board card yet).
        The information set is defined by the acting player's private card and the round-1 history.
        """
        plays = len(history)
        acting = plays % 2  # 0 if player0 is to move, 1 otherwise

        # Terminal test for round 1.
        if self.is_terminal_r1(history):
            if history.endswith('f'):
                # If a fold occurred, evaluate terminal payoff for round 1.
                return self.evaluate_terminal_r1(history, pot, p0_card, p1_card, traversing)
            else:
                # No fold: end of round 1 reached. Reveal board card and start round 2.
                # Choose board card uniformly from the remaining deck (cards[2:] from shuffled deck)
                board = random.choice(self.deck[2:])
                return self.external_cfr_r2(p0_card, p1_card, board, "", pot, traversing, rp0, rp1)

        # Build the information set key.
        if acting == 0:
            infoset = f"R1|{p0_card}|{history}"
        else:
            infoset = f"R1|{p1_card}|{history}"
        if infoset not in self.node_map:
            self.node_map[infoset] = Node(infoset)
        node = self.node_map[infoset]
        strategy = node.get_strategy(rp0 if acting == 0 else rp1)

        # If it is the traversing player's turn, average over all actions.
        if acting == traversing:
            util = np.zeros(NUM_ACTIONS)
            node_util = 0.0
            for a in range(NUM_ACTIONS):
                next_history = history + ACTIONS[a]
                next_pot = pot
                if ACTIONS[a] == 'b':
                    next_pot += self.bet_r1
                child_util = self.external_cfr_r1(p0_card, p1_card, next_history, next_pot, traversing,
                                                  rp0 * (strategy[a] if acting == 0 else 1),
                                                  rp1 * (strategy[a] if acting == 1 else 1))
                util[a] = child_util
                node_util += strategy[a] * util[a]
            # Regret update uses the opponent’s reach probability.
            opp_rp = rp1 if acting == 0 else rp0
            for a in range(NUM_ACTIONS):
                regret = util[a] - node_util
                node.regret_sum[a] += opp_rp * regret
            return node_util
        else:
            # On the opponent’s turn, sample an action according to strategy.
            node.strategy_sum += strategy
            a = np.random.choice(NUM_ACTIONS, p=strategy)
            next_history = history + ACTIONS[a]
            next_pot = pot
            if ACTIONS[a] == 'b':
                next_pot += self.bet_r1
            return self.external_cfr_r1(p0_card, p1_card, next_history, next_pot, traversing,
                                        rp0 * (strategy[a] if acting == 0 else 1),
                                        rp1 * (strategy[a] if acting == 1 else 1))

    def external_cfr_r2(self, p0_card, p1_card, board, history, pot, traversing, rp0, rp1):
        """
        External CFR recursion for round 2 (after the board card is revealed).
        The information set key now includes the board card.
        """
        plays = len(history)
        acting = plays % 2

        if self.is_terminal_r2(history):
            return self.evaluate_terminal_r2(history, pot, p0_card, p1_card, board, traversing)

        if acting == 0:
            infoset = f"R2|{p0_card}|{board}|{history}"
        else:
            infoset = f"R2|{p1_card}|{board}|{history}"
        if infoset not in self.node_map:
            self.node_map[infoset] = Node(infoset)
        node = self.node_map[infoset]
        strategy = node.get_strategy(rp0 if acting == 0 else rp1)

        if acting == traversing:
            util = np.zeros(NUM_ACTIONS)
            node_util = 0.0
            for a in range(NUM_ACTIONS):
                next_history = history + ACTIONS[a]
                next_pot = pot
                if ACTIONS[a] == 'b':
                    next_pot += self.bet_r2
                child_util = self.external_cfr_r2(p0_card, p1_card, board, next_history, next_pot, traversing,
                                                  rp0 * (strategy[a] if acting == 0 else 1),
                                                  rp1 * (strategy[a] if acting == 1 else 1))
                util[a] = child_util
                node_util += strategy[a] * util[a]
            opp_rp = rp1 if acting == 0 else rp0
            for a in range(NUM_ACTIONS):
                regret = util[a] - node_util
                node.regret_sum[a] += opp_rp * regret
            return node_util
        else:
            node.strategy_sum += strategy
            a = np.random.choice(NUM_ACTIONS, p=strategy)
            next_history = history + ACTIONS[a]
            next_pot = pot
            if ACTIONS[a] == 'b':
                next_pot += self.bet_r2
            return self.external_cfr_r2(p0_card, p1_card, board, next_history, next_pot, traversing,
                                        rp0 * (strategy[a] if acting == 0 else 1),
                                        rp1 * (strategy[a] if acting == 1 else 1))

    # --- Terminal conditions and evaluations ---
    def is_terminal_r1(self, history):
        # In round 1, any fold terminates immediately.
        if history.endswith('f'):
            return True
        # For simplicity, assume round 1 ends when both players have acted.
        if len(history) >= 2:
            return True
        return False

    def is_terminal_r2(self, history):
        # In round 2, any fold terminates.
        if history.endswith('f'):
            return True
        # Otherwise, let round 2 end when both players have acted.
        if len(history) >= 2:
            return True
        return False

    def evaluate_terminal_r1(self, history, pot, p0_card, p1_card, traversing):
        # If a fold occurred in round 1, then the player who took the last action loses.
        folded = (len(history) - 1) % 2
        winner = 1 - folded
        return pot if winner == traversing else -pot

    def evaluate_terminal_r2(self, history, pot, p0_card, p1_card, board, traversing):
        if history.endswith('f'):
            folded = (len(history) - 1) % 2
            winner = 1 - folded
            return pot if winner == traversing else -pot
        # Otherwise, at showdown determine the winner.
        # In Leduc, a hand’s strength is determined by pairing with the board:
        # If a private card matches the board’s rank, treat it as a pair (which beats any single card).
        def hand_strength(card, board):
            if card // 2 == board // 2:
                return 100 + card  # Give a large bonus for a pair
            else:
                return card
        p0_strength = hand_strength(p0_card, board)
        p1_strength = hand_strength(p1_card, board)
        winner = 0 if p0_strength > p1_strength else 1
        return pot if winner == traversing else -pot

# --- Running the algorithm ---
if __name__ == "__main__":
    start_time = time.time()
    iterations = 100000
    leduc = LeducCFR(iterations)
    avg_util = leduc.cfr()
    print(f"Average utility after {iterations} CFR iterations: {avg_util}")
    print("Elapsed time:", time.time() - start_time, "seconds")
