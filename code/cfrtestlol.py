#!/usr/bin/env python3
"""
cfr_trainer.py

This script implements a CFR trainer for Leduc Hold’Em that works with the provided game logic.
It uses alternating updates (updating one player’s regrets per iteration) and linear averaging 
to compute the average (approximate equilibrium) strategy for each information set.

Assumptions:
  • The game logic (classes Card, LeducGameState, LeducHoldemGame) is in a separate module,
    for example, in a file named "leduc_holdem.py".
  • Do not modify the game logic.
"""

import random
import copy
from leduc_holdem import LeducHoldemGame, LeducGameState  # import the game classes

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def clone_game(game):
    """
    Creates a deep copy of a LeducHoldemGame instance.
    This is necessary so that when we “simulate” an action in the CFR recursion,
    we do not modify the original game state.
    """
    new_game = LeducHoldemGame.__new__(LeducHoldemGame)
    new_game.ranks = game.ranks
    new_game.suits = game.suits
    new_game.bet_amounts = game.bet_amounts
    new_game.max_bets = game.max_bets
    
    # Clone the game state:
    new_state = LeducGameState.__new__(LeducGameState)
    s = game.state
    new_state.round = s.round
    new_state.deck = s.deck.copy()   # shallow copy is enough for our purpose
    new_state.private_cards = s.private_cards.copy()
    new_state.board = s.board        # board card (Card) – treated as immutable here
    new_state.pot = s.pot
    new_state.contributions = s.contributions.copy()
    new_state.total_contributions = s.total_contributions.copy()
    new_state.bet_count = s.bet_count
    new_state.current_player = s.current_player
    new_state.last_action = s.last_action.copy()
    new_state.terminal = s.terminal
    new_state.winner = s.winner
    new_game.state = new_state
    return new_game

def get_info_set_key(game, player):
    """
    Returns a string that uniquely identifies an information set for the specified player.
    
    In imperfect-information games (like poker), an information set hides parts of the true state.
    Here we include:
      - The acting player’s private card,
      - The board card (if dealt),
      - The current round number,
      - And the betting history (actions taken in the current round).
      
    For example, an info set might look like "JH|QH|1|bet-check" meaning that the player
    holds Jack of Hearts, no board card (or if in round 2 the board card would appear), round 1,
    and the betting history so far is “bet” then “check”.
    """
    s = game.state
    # Only include the acting player’s private card (do not reveal the opponent’s card)
    private_card = s.private_cards[player]
    card_str = str(private_card) if private_card is not None else "None"
    # Include board card if it exists
    board_str = str(s.board) if s.board is not None else ""
    # Round number as string
    round_str = str(s.round)
    # Betting history in the current round (a list of actions)
    history_str = "-".join(s.last_action) if s.last_action else ""
    # Combine pieces into a key (separated by '|')
    info_set = f"{card_str}|{board_str}|{round_str}|{history_str}"
    return info_set

# -------------------------------------------------------------------
# CFR Trainer Class
# -------------------------------------------------------------------

class CFRTrainer:
    def __init__(self, iterations):
        """
        Initializes a trainer for CFR with a specified number of iterations.
        
        It holds dictionaries for:
          • regretSum: cumulative counterfactual regret values for each info set.
          • strategySum: cumulative strategy (probability weighting) for each info set.
        """
        self.iterations = iterations
        self.regretSum = {}    # {info_set_key: {action: cumulative regret}}
        self.strategySum = {}  # {info_set_key: {action: cumulative probability weight}}

    def get_strategy(self, info_set, legal_actions):
        """
        Computes the current strategy for an information set via regret matching.

        If there are positive cumulative regrets for some actions, the strategy
        selects actions in proportion to those positive regrets; otherwise, a uniform distribution is used.
        """
        if info_set not in self.regretSum:
            self.regretSum[info_set] = {a: 0.0 for a in legal_actions}
        regrets = self.regretSum[info_set]
        # Only consider positive regrets
        positive_regrets = {a: regrets[a] if regrets[a] > 0 else 0 for a in legal_actions}
        total_positive = sum(positive_regrets.values())
        strategy = {}
        if total_positive > 0:
            for a in legal_actions:
                strategy[a] = positive_regrets[a] / total_positive
        else:
            # If all regrets are zero or negative, use a uniform strategy.
            for a in legal_actions:
                strategy[a] = 1.0 / len(legal_actions)
        return strategy

    def cfr(self, game, updating_player, reach_upd, reach_opp):
        """
        The recursive CFR routine.
        
        Parameters:
          game           : the current LeducHoldemGame instance (state in the game tree).
          updating_player: the player index (0 or 1) whose regrets we update this iteration.
          reach_upd      : the probability of reaching this node from the updating player's own decisions.
          reach_opp      : the probability of reaching this node from the opponent's actions.
          
        Returns:
          The expected utility (payoff) for the updating_player from the current game state.
        """
        # Terminal check: if the game is over, return the payoff for the updating player.
        if game.is_terminal():
            payoffs = game.showdown()  # payoffs is a two-element list: [payoff_player0, payoff_player1]
            return payoffs[updating_player]

        legal_actions = game.get_legal_actions()

        # Handle chance node: In our game logic, when the board has not yet been dealt (in round 2),
        # the only legal action is 'deal_board'.
        if legal_actions == ['deal_board']:
            game.apply_action('deal_board')
            return self.cfr(game, updating_player, reach_upd, reach_opp)

        current_player = game.state.current_player
        info_set = get_info_set_key(game, current_player)

        if current_player == updating_player:
            # Decision node for the updating player: compute current strategy using regret matching.
            strategy = self.get_strategy(info_set, legal_actions)
            node_value = 0.0
            action_utilities = {}
            # For each legal action, simulate (by cloning the state) and recursively compute the utility.
            for action in legal_actions:
                next_game = clone_game(game)
                next_game.apply_action(action)
                # Multiply the updating player's reach probability by the probability of taking this action.
                action_utilities[action] = self.cfr(next_game, updating_player, reach_upd * strategy[action], reach_opp)
                node_value += strategy[action] * action_utilities[action]

            # For averaging the strategy later, initialize the strategy sum if needed.
            if info_set not in self.strategySum:
                self.strategySum[info_set] = {a: 0.0 for a in legal_actions}
            # Update regret sums and strategy sums for each action.
            for action in legal_actions:
                # The counterfactual regret for an action is the difference between the
                # utility from taking that action and the average node value.
                regret = action_utilities[action] - node_value
                # Regret is weighted by the probability that the opponent reached this node.
                self.regretSum[info_set][action] += reach_opp * regret
                # For linear averaging, add the product of current strategy and the updating player's reach probability.
                self.strategySum[info_set][action] += reach_upd * strategy[action]
            return node_value
        else:
            # At an opponent node, assume the opponent plays according to their current strategy.
            # (No regret updates are done at the opponent’s decision nodes during this iteration.)
            strategy = self.get_strategy(info_set, legal_actions)
            node_value = 0.0
            for action in legal_actions:
                next_game = clone_game(game)
                next_game.apply_action(action)
                # Multiply the opponent's reach probability by the probability of the chosen action.
                node_value += strategy[action] * self.cfr(next_game, updating_player, reach_upd, reach_opp * strategy[action])
            return node_value

    def train(self):
        """
        Runs the CFR training process for a specified number of iterations.
        Alternating updates: in iteration i, only player (i mod 2) has its regrets updated.
        """
        
        util = 0.0
        util_p0 = 0.0
        util_p1 = 0.0
        for i in range(self.iterations):
            # Create a new game instance at the start of each iteration.
            game = LeducHoldemGame()
            # Determine which player to update this iteration (alternating updates).
            updating_player = i % 2
            # Start the CFR recursion at the root of the game tree with reach probabilities 1.
            value = self.cfr(game, updating_player, 1.0, 1.0)
            util +=value
            if updating_player == 0:
                util_p0+=value
            else:
                util_p1+=value
            
            # Optional: print progress every 10% of iterations.
            if (i + 1) % (self.iterations // 10) == 0:
                print(f"Iteration {i + 1}/{self.iterations} completed.")
        print("Average game utility (per iteration):", util / self.iterations)
        print("Expected utility for player 0 (first mover):", util_p0 / (self.iterations / 2))
        print("Expected utility for player 1 (second mover):", util_p1 / (self.iterations / 2))

    def get_average_strategy(self):
        """
        Computes the overall average strategy for each information set from the cumulative strategy sums.
        
        Returns:
          A dictionary mapping each info set key (string) to a dictionary of {action: probability}.
        """
        avg_strategy = {}
        for info_set, strategy_sum in self.strategySum.items():
            total = sum(strategy_sum.values())
            if total > 0:
                avg_strategy[info_set] = {action: strategy_sum[action] / total for action in strategy_sum}
            else:
                num_actions = len(strategy_sum)
                avg_strategy[info_set] = {action: 1.0 / num_actions for action in strategy_sum}
        return avg_strategy

# -------------------------------------------------------------------
# Main: Run Training and Print Average Strategy
# -------------------------------------------------------------------

if __name__ == '__main__':
    # You can adjust the number of iterations as needed.
    iterations = 900000
    trainer = CFRTrainer(iterations)
    trainer.train()
    
    avg_strategy = trainer.get_average_strategy()
    print("\nComputed Average Strategy per Information Set:")
    for info_set, strategy in avg_strategy.items():
        print(f"{info_set}: {strategy}")

    def simulate_game(avg_strategy, num_simulations=10000):
        total_payoffs = [0.0, 0.0]
        for _ in range(num_simulations):
            game = LeducHoldemGame()
            while not game.is_terminal():
                legal_actions = game.get_legal_actions()
                # Handle chance node: if the only action is 'deal_board', apply it.
                if legal_actions == ['deal_board']:
                    game.apply_action('deal_board')
                    continue
                current_player = game.state.current_player
                info_set = get_info_set_key(game, current_player)
                # If the information set was never reached during training, use a uniform strategy.
                if info_set in avg_strategy:
                    strat = avg_strategy[info_set]
                    actions = list(strat.keys())
                    probs = [strat[a] for a in actions]
                else:
                    actions = legal_actions
                    probs = [1.0 / len(legal_actions)] * len(legal_actions)
                chosen_action = random.choices(actions, weights=probs, k=1)[0]
                game.apply_action(chosen_action)
            payoffs = game.showdown()
            total_payoffs[0] += payoffs[0]
            total_payoffs[1] += payoffs[1]
        return [p / num_simulations for p in total_payoffs]

    # ---- New Code Start ----
    # Evaluate the average strategy by simulating games.
    num_simulations = 900000  # Adjust this number to increase accuracy if needed.
    expected_values = simulate_game(trainer.get_average_strategy(), num_simulations)
    print("\nExpected game value under the average strategy:")
    print("Player 0 (first mover):", expected_values[0])
    print("Player 1 (second mover):", expected_values[1])
    # ---- New Code End ----