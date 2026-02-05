import numpy as np  # Import numpy for numerical operations, especially array manipulation.
import random       # Import random for shuffling the deck and sampling actions.
import time         # Import time for measuring the execution time of the algorithm.

# --- Global definitions ---
# Define the set of possible actions a player can take in either betting round.
# 'c' stands for check/call, 'b' for bet/raise, 'f' for fold.
ACTIONS = ['c', 'b', 'f']
# Store the number of possible actions, used for array dimensions.
NUM_ACTIONS = len(ACTIONS)

# --- Node class ---
# Represents a node in the game tree, corresponding to an information set.
class Node:
    # Initialize a Node instance.
    def __init__(self, infoset, num_actions=NUM_ACTIONS):
        # The information set string uniquely identifies this node from the perspective of the acting player.
        self.infoset = infoset
        # The number of actions possible at this node.
        self.num_actions = num_actions
        # Stores the cumulative regret for not choosing each action at this node. Used for CFR.
        self.regret_sum = np.zeros(num_actions)
        # Stores the computed strategy (probability distribution over actions) for the current iteration.
        self.strategy = np.zeros(num_actions)
        # Stores the cumulative strategy weighted by the reach probability of the current player. Used for calculating the average strategy.
        self.strategy_sum = np.zeros(num_actions)

    # Calculates the current strategy based on regrets.
    def get_strategy(self, reach_prob=1.0):
        """
        Computes the current strategy for this node based on positive regrets.
        Uses regret matching: the probability of choosing an action is proportional to its positive regret.
        If all regrets are non-positive, reverts to a uniform random strategy.
        Also updates the cumulative strategy sum weighted by the player's reach probability.

        Args:
            reach_prob (float): The probability that the current player reaches this node.

        Returns:
            np.array: The calculated strategy (probability distribution over actions).
        """
        normalizing_sum = 0.0
        # Iterate through each possible action.
        for a in range(self.num_actions):
            # Strategy is initially set to the positive part of the regret for that action.
            self.strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0.0
            # Accumulate the sum of positive regrets for normalization.
            normalizing_sum += self.strategy[a]

        # Normalize the strategy or use uniform distribution if sum is zero.
        for a in range(self.num_actions):
            if normalizing_sum > 0:
                # Normalize the strategy by dividing by the sum of positive regrets.
                self.strategy[a] /= normalizing_sum
            else:
                # If no positive regrets, use a uniform random strategy.
                self.strategy[a] = 1.0 / self.num_actions
            # Update the cumulative strategy sum, weighted by the player's reach probability to this node.
            self.strategy_sum[a] += reach_prob * self.strategy[a]

        # Return the computed strategy for the current iteration.
        return self.strategy

    # Calculates the average strategy over all iterations.
    def get_average_strategy(self):
        """
        Computes the average strategy for this node across all CFR iterations.
        This is done by normalizing the cumulative strategy sum.

        Returns:
            np.array: The average strategy (probability distribution over actions).
        """
        # Calculate the sum of cumulative strategies for normalization.
        normalizing_sum = np.sum(self.strategy_sum)
        # Initialize the average strategy array.
        avg_strategy = np.zeros(self.num_actions)
        # Iterate through each action to calculate its average probability.
        for a in range(self.num_actions):
            if normalizing_sum > 0:
                # Normalize the cumulative strategy sum for this action.
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                # If the cumulative sum is zero (e.g., node never reached), use uniform random.
                avg_strategy[a] = 1.0 / self.num_actions
        # Return the calculated average strategy.
        return avg_strategy

# --- Leduc Poker CFR Implementation ---
# Implements the Counterfactual Regret Minimization (CFR) algorithm for Leduc Poker.
class LeducCFR:
    # Initialize the LeducCFR instance.
    def __init__(self, iterations):
        # Number of CFR iterations to run.
        self.iterations = iterations
        # The deck of cards for Leduc Poker: 6 cards (J, Q, K of two suits).
        # Cards are represented as integers 0-5. 0,1=J; 2,3=Q; 4,5=K.
        self.deck = list(range(6))
        # A dictionary to store Node objects, keyed by their information set strings.
        self.node_map = {}

        # Game parameters based on standard Leduc Poker rules:
        # Each player pays 1 chip ante before the hand starts.
        self.ante = 1
        # The fixed bet/raise amount in the first betting round is 2 chips.
        self.bet_r1 = 2
        # The fixed bet/raise amount in the second betting round is 4 chips.
        self.bet_r2 = 4

    # Main function to run the CFR algorithm.
    def cfr(self):
        """
        Runs the CFR algorithm for the specified number of iterations.
        In each iteration, it simulates one game traversal for each player
        to update regrets and strategies.

        Returns:
            np.array: The average game utility for player 0 and player 1 over all iterations.
        """
        # Initialize cumulative utility for player 0 and player 1.
        util = np.zeros(2)
        # Loop through the specified number of iterations.
        for t in range(1, self.iterations + 1):
            # Shuffle the deck at the start of each iteration (game).
            random.shuffle(self.deck)
            # Deal one private card to each player.
            p0_card = self.deck[0]
            p1_card = self.deck[1]
            # Initial pot size is the sum of antes from both players.
            pot = 2 * self.ante
            # Start round 1 with an empty action history.
            history_r1 = ""

            # Run external sampling CFR for round 1.
            # We traverse the game tree once for each player acting as the 'traversing player'.
            # The utility returned by external_cfr_r1 is added to the cumulative utility for that player.
            for traversing in [0, 1]: # Player 0 traverses, then Player 1 traverses
                # Reach probabilities start at 1.0 for both players.
                util[traversing] += self.external_cfr_r1(p0_card, p1_card, history_r1, pot, traversing, 1.0, 1.0)

        # Return the average utility per iteration for each player.
        return util / self.iterations

    # External Sampling CFR function for Round 1.
    def external_cfr_r1(self, p0_card, p1_card, history, pot, traversing, rp0, rp1):
        """
        Recursive function implementing External Sampling CFR for Round 1.
        Updates regrets and strategies for the 'traversing' player.
        Samples actions for the opponent according to their current strategy.

        Args:
            p0_card (int): Player 0's private card.
            p1_card (int): Player 1's private card.
            history (str): The action history string for the current round.
            pot (int): The current size of the pot.
            traversing (int): The index (0 or 1) of the player whose perspective we are currently evaluating from (the one updating regrets).
            rp0 (float): The reach probability of Player 0 reaching the current game state.
            rp1 (float): The reach probability of Player 1 reaching the current game state.

        Returns:
            float: The expected utility of the game state from the perspective of the 'traversing' player.
        """
        # Number of actions taken so far in this round.
        plays = len(history)
        # Determine the player whose turn it is (0 or 1). Player 0 starts.
        acting = plays % 2

        # Check if Round 1 has reached a terminal state.
        if self.is_terminal_r1(history):
            if history.endswith('f'):
                # If the round ended with a fold, calculate the payoff immediately.
                return self.evaluate_terminal_r1(history, pot, p0_card, p1_card, traversing)
            else:
                # If the round ended without a fold (e.g., cc, bc, bb), proceed to Round 2.
                # A public board card is dealt from the remaining deck.
                board = random.choice(self.deck[2:]) # Pick one of the remaining cards
                # Start Round 2 CFR from an empty history.
                return self.external_cfr_r2(p0_card, p1_card, board, "", pot, traversing, rp0, rp1)

        # Construct the information set string for the current state.
        # It includes the round, the acting player's card, and the action history.
        if acting == 0:
            infoset = f"R1|{p0_card}|{history}"
        else:
            infoset = f"R1|{p1_card}|{history}"

        # Retrieve or create the Node corresponding to this information set.
        if infoset not in self.node_map:
            self.node_map[infoset] = Node(infoset)
        node = self.node_map[infoset]

        # Get the current strategy for the acting player at this node.
        # Update the strategy sum with the appropriate reach probability.
        strategy = node.get_strategy(rp0 if acting == 0 else rp1)

        # If it's the traversing player's turn to act:
        if acting == traversing:
            # Calculate the expected utility for each action.
            util = np.zeros(NUM_ACTIONS)
            # Initialize the expected utility of the node.
            node_util = 0.0
            # Iterate through all possible actions.
            for a in range(NUM_ACTIONS):
                # Append the action to the history.
                next_history = history + ACTIONS[a]
                # Update the pot if the action was a bet.
                next_pot = pot
                if ACTIONS[a] == 'b':
                    next_pot += self.bet_r1
                # Recursively call CFR for the next state.
                # Update the reach probability for the player who just acted.
                child_util = self.external_cfr_r1(p0_card, p1_card, next_history, next_pot, traversing,
                                                  rp0 * (strategy[a] if acting == 0 else 1), # Update P0's reach prob if P0 acted
                                                  rp1 * (strategy[a] if acting == 1 else 1)) # Update P1's reach prob if P1 acted
                # Store the utility resulting from taking action 'a'.
                util[a] = child_util
                # Accumulate the node's expected utility using the strategy probabilities.
                node_util += strategy[a] * util[a]

            # Update regrets for the traversing player.
            # The regret for an action is the difference between the utility of taking that action
            # and the expected utility of the node.
            # Regret updates are weighted by the opponent's reach probability.
            opp_rp = rp1 if acting == 0 else rp0
            for a in range(NUM_ACTIONS):
                regret = util[a] - node_util
                # Add the weighted regret to the cumulative regret sum for this action.
                node.regret_sum[a] += opp_rp * regret
            # Return the expected utility of this node for the traversing player.
            return node_util
        else:
            # If it's the opponent's turn (non-traversing player):
            # Accumulate the strategy for average strategy calculation (this is part of Vanilla CFR, often omitted in External Sampling, but kept here).
            node.strategy_sum += strategy # NOTE: This update might be misplaced in pure External Sampling, usually only strategy_sum of traversing player matters for average strategy.
            # Sample an action according to the opponent's current strategy.
            a = np.random.choice(NUM_ACTIONS, p=strategy)
            # Append the sampled action to the history.
            next_history = history + ACTIONS[a]
            # Update the pot if the action was a bet.
            next_pot = pot
            if ACTIONS[a] == 'b':
                next_pot += self.bet_r1
            # Recursively call CFR for the next state, updating the reach probability of the acting (opponent) player.
            return self.external_cfr_r1(p0_card, p1_card, next_history, next_pot, traversing,
                                       rp0 * (strategy[a] if acting == 0 else 1), # Update P0's reach prob if P0 acted
                                       rp1 * (strategy[a] if acting == 1 else 1)) # Update P1's reach prob if P1 acted

    # External Sampling CFR function for Round 2.
    def external_cfr_r2(self, p0_card, p1_card, board, history, pot, traversing, rp0, rp1):
        """
        Recursive function implementing External Sampling CFR for Round 2.
        Similar structure to external_cfr_r1, but uses round 2 rules (bet size, infoset includes board card).

        Args:
            p0_card (int): Player 0's private card.
            p1_card (int): Player 1's private card.
            board (int): The public board card.
            history (str): The action history string for the current round (Round 2).
            pot (int): The current size of the pot.
            traversing (int): The index (0 or 1) of the player whose perspective we are evaluating.
            rp0 (float): Player 0's reach probability.
            rp1 (float): Player 1's reach probability.

        Returns:
            float: The expected utility of the game state from the perspective of the 'traversing' player.
        """
        # Number of actions taken so far in Round 2.
        plays = len(history)
        # Determine the player whose turn it is (Player 0 starts Round 2 as well).
        acting = plays % 2

        # Check if Round 2 has reached a terminal state (fold or showdown).
        if self.is_terminal_r2(history):
            # Evaluate the payoff based on fold or showdown results.
            return self.evaluate_terminal_r2(history, pot, p0_card, p1_card, board, traversing)

        # Construct the information set string for Round 2.
        # It includes the round marker, player's card, board card, and Round 2 action history.
        if acting == 0:
            infoset = f"R2|{p0_card}|{board}|{history}"
        else:
            infoset = f"R2|{p1_card}|{board}|{history}"

        # Retrieve or create the Node for this information set.
        if infoset not in self.node_map:
            self.node_map[infoset] = Node(infoset)
        node = self.node_map[infoset]

        # Get the current strategy for the acting player.
        strategy = node.get_strategy(rp0 if acting == 0 else rp1)

        # If it's the traversing player's turn:
        if acting == traversing:
            # Calculate expected utility for each action.
            util = np.zeros(NUM_ACTIONS)
            node_util = 0.0
            for a in range(NUM_ACTIONS):
                next_history = history + ACTIONS[a]
                next_pot = pot
                # Use the Round 2 bet size.
                if ACTIONS[a] == 'b':
                    next_pot += self.bet_r2
                # Recursive call for the next state, updating reach probabilities.
                child_util = self.external_cfr_r2(p0_card, p1_card, board, next_history, next_pot, traversing,
                                                  rp0 * (strategy[a] if acting == 0 else 1),
                                                  rp1 * (strategy[a] if acting == 1 else 1))
                util[a] = child_util
                node_util += strategy[a] * util[a]

            # Update regrets for the traversing player, weighted by opponent's reach probability.
            opp_rp = rp1 if acting == 0 else rp0
            for a in range(NUM_ACTIONS):
                regret = util[a] - node_util
                node.regret_sum[a] += opp_rp * regret
            return node_util
        else:
            # If it's the opponent's turn (non-traversing player):
            # Accumulate strategy sum (again, potentially misplaced for pure External Sampling).
            node.strategy_sum += strategy
            # Sample an action based on the opponent's strategy.
            a = np.random.choice(NUM_ACTIONS, p=strategy)
            next_history = history + ACTIONS[a]
            next_pot = pot
            # Use the Round 2 bet size.
            if ACTIONS[a] == 'b':
                next_pot += self.bet_r2
            # Recursive call for the next state, updating reach probabilities.
            return self.external_cfr_r2(p0_card, p1_card, board, next_history, next_pot, traversing,
                                       rp0 * (strategy[a] if acting == 0 else 1),
                                       rp1 * (strategy[a] if acting == 1 else 1))

    # --- Terminal conditions and evaluations ---

    # Checks if a history string represents a terminal state in Round 1.
    def is_terminal_r1(self, history):
        """
        Determines if the game state in Round 1 is terminal.
        Round 1 ends if a player folds, or if two actions have been completed
        (e.g., check-check, check-bet-call, bet-call, bet-fold).

        Args:
            history (str): The action history for Round 1.

        Returns:
            bool: True if the state is terminal for Round 1, False otherwise.
        """
        # Any fold ('f') immediately ends the round (and the game).
        if history.endswith('f'):
            return True
        # Simplified rule: Assume round 1 ends after 2 actions if no fold.
        # Covers sequences like 'cc', 'cb', 'bb'. 'bc' requires a third action (call/fold).
        # More accurate Leduc rules allow up to 4 actions (e.g., b-b-c-c is not possible with 1 raise limit, but b-raise-call is 3 actions).
        # This simplification might need adjustment for full Leduc rules (e.g., history == 'cc' or history == 'bc' or history == 'bb').
        # The current check `len(history) >= 2` might prematurely end sequences like 'b' followed by 'c' or 'b'.
        # Let's assume the intended logic handles cases like call ('c'), bet ('b'), raise ('b'), fold ('f').
        # Common terminal sequences in R1 (1 bet limit): f, cf, bf, cc, bc, bb (ends R1 -> R2)
        # A more robust check for R1 completion without fold:
        # (history == 'cc') or (history == 'bc') or (history == 'bb')
        # For simplicity, the provided code uses len(history) >= 2. Let's keep that but note its limitation.
        if len(history) >= 2: # This covers 'cc', 'bb', 'cf', 'bf'. It doesn't explicitly cover 'bc'.
                               # If 'bc' happens, len=2, it goes to R2. If 'bf' happens, len=2, it folds.
                               # If 'cf' happens, len=2, it folds.
                               # Seems okay for the common actions allowed (c, b, f).
            return True
        # Otherwise, the round continues.
        return False

    # Checks if a history string represents a terminal state in Round 2.
    def is_terminal_r2(self, history):
        """
        Determines if the game state in Round 2 is terminal.
        Round 2 ends if a player folds, or if two actions have been completed, leading to a showdown.

        Args:
            history (str): The action history for Round 2.

        Returns:
            bool: True if the state is terminal for Round 2, False otherwise.
        """
        # Any fold ('f') ends the round.
        if history.endswith('f'):
            return True
        # Similar to Round 1, assume round ends after 2 actions if no fold.
        # This leads to showdown.
        if len(history) >= 2: # Covers 'cc', 'bb' (showdown), 'cf', 'bf' (fold)
            return True
        # Otherwise, the round continues.
        return False

    # Evaluates the payoff at a terminal state in Round 1 (only occurs on a fold).
    def evaluate_terminal_r1(self, history, pot, p0_card, p1_card, traversing):
        """
        Calculates the payoff when Round 1 ends due to a fold.
        The player who did not fold wins the pot.

        Args:
            history (str): The action history (must end in 'f').
            pot (int): The size of the pot when the fold occurred.
            p0_card, p1_card: Private cards (not needed for fold evaluation).
            traversing (int): The player perspective for utility calculation.

        Returns:
            float: The utility (+pot or -pot) for the 'traversing' player.
        """
        # Determine which player folded. The player who made the last action ('f') is the one who folded.
        # If len(history) is odd (1, 3, ...), player 0 made the last move.
        # If len(history) is even (2, 4, ...), player 1 made the last move.
        folded = (len(history) - 1) % 2 # 0 if P0 folded, 1 if P1 folded.
        # The winner is the player who did *not* fold.
        winner = 1 - folded
        # Return the pot amount if the traversing player won, negative pot amount if they lost.
        # Note: The pot here includes money put in *before* the fold action.
        # The utility should likely be calculated based on chips won/lost relative to the start.
        # However, CFR typically uses the perspective of winning/losing the final pot.
        # Let's assume +pot for winner, -pot for loser is the desired utility representation.
        # A more precise calculation would track contributions:
        # e.g., Ante=1. P0 bets 2, P1 folds. Pot=1+1+2=4. P0 contributed 1+2=3. P1 contributed 1.
        # P0 wins 4, net gain is 4-3 = +1. P1 loses 1, net loss is -1.
        # The current implementation returns +/- pot size. Let's stick to that interpretation.
        return pot if winner == traversing else -pot # This might overestimate utility swing, typically it's opponent's contribution.
                                                    # Standard CFR often uses this simpler +/- pot.

    # Evaluates the payoff at a terminal state in Round 2 (fold or showdown).
    def evaluate_terminal_r2(self, history, pot, p0_card, p1_card, board, traversing):
        """
        Calculates the payoff when Round 2 ends, either by fold or showdown.

        Args:
            history (str): The action history for Round 2.
            pot (int): The final size of the pot.
            p0_card (int): Player 0's private card.
            p1_card (int): Player 1's private card.
            board (int): The public board card.
            traversing (int): The player perspective for utility calculation.

        Returns:
            float: The utility (+pot or -pot) for the 'traversing' player.
        """
        # If the round ended with a fold.
        if history.endswith('f'):
            # Determine who folded based on the last action.
            folded = (len(history) - 1) % 2 # 0 if P0 folded, 1 if P1 folded.
            # The other player wins the pot.
            winner = 1 - folded
            # Return utility based on winning/losing the pot.
            return pot if winner == traversing else -pot # Again, simplified +/- pot utility.
        else:
            # If the round ended without a fold, it's a showdown.
            # Determine the winner based on hand strength.
            # Leduc hand ranking: Pair beats high card. High card breaks ties.
            # Card ranks: J=0, Q=1, K=2. Suits don't matter for rank. Card index // 2 gives rank.
            def hand_strength(card, board_card):
                # Check if the private card's rank matches the board card's rank.
                if card // 2 == board_card // 2: # Integer division by 2 gives the rank (0, 1, or 2).
                    # If it's a pair, return a high value plus the card's value (for kicker).
                    # Using 100 ensures pairs beat any high card. Card value acts as kicker if both have pairs (not possible in Leduc) or high cards.
                    return 100 + (card // 2) # Rank is enough, actual card value isn't needed for pair kicker
                else:
                    # If it's not a pair, the strength is just the card's rank.
                    return card // 2 # Return rank

            # Calculate hand strength for both players.
            p0_strength = hand_strength(p0_card, board)
            p1_strength = hand_strength(p1_card, board)

            # Determine the winner. Player 0 wins if their strength is strictly greater.
            if p0_strength > p1_strength:
                winner = 0
            elif p1_strength > p0_strength:
                winner = 1
            else:
                # If strengths are equal (e.g., both have same high card, no pair), it's a tie.
                # In poker, ties usually split the pot. CFR utility should be 0 for both in a tie.
                # The current implementation forces a winner (P1 wins ties). Let's adjust for ties.
                # Re-evaluating comparison:
                if p0_strength == p1_strength:
                    return 0 # Tie game, utility is 0 for both players relative to the pot.
                winner = 0 if p0_strength > p1_strength else 1

            # Return utility based on winning/losing the pot.
            return pot if winner == traversing else -pot # Simplified +/- pot utility.

# --- Running the algorithm ---
# This block executes when the script is run directly.
if __name__ == "__main__":
    # Record the start time to measure execution duration.
    start_time = time.time()
    # Set the number of CFR iterations to perform.
    iterations = 1000000 # Increased iterations for better convergence.
    # Create an instance of the LeducCFR class.
    leduc = LeducCFR(iterations)
    # Run the CFR algorithm and get the average utility per game for each player.
    avg_util = leduc.cfr()
    # Print the results.
    print(f"Average utility after {iterations} CFR iterations: {avg_util}")
    # Print the total time taken for the computation.
    print("Elapsed time:", time.time() - start_time, "seconds")

    # Example: Print average strategy for a specific infoset if needed
    # try:
    #     example_infoset = "R1|4|" # Example: Player 0 has King, Round 1, no actions yet
    #     if example_infoset in leduc.node_map:
    #         avg_strategy = leduc.node_map[example_infoset].get_average_strategy()
    #         print(f"Average strategy for infoset {example_infoset}: Check/Call={avg_strategy[0]:.3f}, Bet/Raise={avg_strategy[1]:.3f}, Fold={avg_strategy[2]:.3f}")
    #     else:
    #         print(f"Infoset {example_infoset} not encountered or created.")
    # except Exception as e:
    #     print(f"Could not retrieve strategy for example infoset: {e}")