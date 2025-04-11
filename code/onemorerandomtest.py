import time

# Game constants for Leduc Hold'em
NUM_RANKS = 3         # Three ranks (e.g., 0,1,2 for J, Q, K)
CARDS_PER_RANK = 2    # Two suits per rank
DECK_SIZE = NUM_RANKS * CARDS_PER_RANK  # Total 6 cards
BET_SIZE_ROUND1 = 2   # Bet amount in round 1
BET_SIZE_ROUND2 = 4   # Bet amount in round 2
MAX_RAISES_PER_ROUND = 1  # Maximum one raise per round (two bets total per round)
ANTE = 1  # Each player antes 1 chip at start

# Data structures for CFR
regrets = {}        # Cumulative regret for each info set: info_key -> [regret_for_each_action]
strategy_sum = {}   # Cumulative strategy probabilities for averaging: info_key -> [sum_probs_for_actions]

# Variables to control which player updates regrets this iteration
update_player = None
current_iter = 0

def info_key(player, private_card, public_card, history):
    """Construct an information set key for the given state."""
    priv_rank = private_card // CARDS_PER_RANK
    pub_rank = "None" if public_card is None else str(public_card // CARDS_PER_RANK)
    return f"p{player}:{priv_rank}:{pub_rank}:{history}"

def regret_matching_strategy(info_str, actions):
    """Return a strategy distribution for the info set using regret-matching."""
    if info_str not in regrets:
        # Initialize regrets and strategy_sum for new info sets
        regrets[info_str] = [0.0] * len(actions)
        strategy_sum[info_str] = [0.0] * len(actions)
    reg = regrets[info_str]
    # Regret matching: positive regrets determine the probabilities
    positive_regs = [r if r > 0 else 0.0 for r in reg]
    total_pos = sum(positive_regs)
    if total_pos > 1e-9:
        return [r / total_pos for r in positive_regs]
    else:
        # If no positive regret, use a uniform strategy
        return [1.0 / len(reg)] * len(reg)

def cfr(current_player, round_index, p0_card, p1_card, public_card,
        p0_contrib, p1_contrib, to_call, raise_count, history,
        reach0, reach1):
    """
    Perform a CFR traversal from the given state.
    Returns (u0, u1): the expected utility for players 0 and 1 from this state.
    """
    # Terminal condition: showdown after both betting rounds
    if round_index > 2:
        # Determine showdown outcome
        pub_rank = public_card // CARDS_PER_RANK
        p0_rank = p0_card // CARDS_PER_RANK
        p1_rank = p1_card // CARDS_PER_RANK
        p0_pair = (p0_rank == pub_rank)
        p1_pair = (p1_rank == pub_rank)
        if p0_pair and not p1_pair:
            # Player 0 has a pair (and player 1 doesn't)
            return (p1_contrib, -p1_contrib)
        elif p1_pair and not p0_pair:
            # Player 1 has a pair
            return (-p0_contrib, p0_contrib)
        else:
            # No pairs (or both have pair, which cannot happen in Leduc): high card wins (or tie)
            if p0_rank > p1_rank:
                return (p1_contrib, -p1_contrib)
            elif p1_rank > p0_rank:
                return (-p0_contrib, p0_contrib)
            else:
                return (0.0, 0.0)  # tie: no net gain
    # Decision node (player action)
    if current_player is not None:
        # List available actions based on the game state
        actions = []
        if to_call > 0:
            # An outstanding bet exists
            actions.append("f")   # fold
            actions.append("c")   # call
            if raise_count < MAX_RAISES_PER_ROUND:
                actions.append("r")   # raise
        else:
            # No bet yet, can either check or bet
            actions.append("ch")  # check
            actions.append("b")   # bet
        # Get info set key and strategy
        info_str = info_key(current_player, p0_card if current_player == 0 else p1_card,
                             public_card, history)
        strategy = regret_matching_strategy(info_str, actions)
        # Accumulate strategy for averaging (linear weight = iteration count)
        global current_iter
        for i, prob in enumerate(strategy):
            strategy_sum[info_str][i] += current_iter * prob
        # Utilities for each action branch and for the node as a whole
        node_util0 = 0.0
        node_util1 = 0.0
        action_utils = []
        # Explore each action
        for idx, action in enumerate(actions):
            # Save state variables to restore after recursion
            prev_p0_contrib, prev_p1_contrib = p0_contrib, p1_contrib
            prev_to_call, prev_raise_count = to_call, raise_count
            prev_history, prev_public, prev_round = history, public_card, round_index
            # Apply the action
            if action == "f":  # Fold
                if current_player == 0:
                    branch_util = (-p0_contrib, p0_contrib)  # P0 loses what they put in
                else:
                    branch_util = (p1_contrib, -p1_contrib)
            elif action == "c":  # Call
                # Match the outstanding bet
                if current_player == 0:
                    p0_contrib += to_call
                else:
                    p1_contrib += to_call
                if round_index == 1:
                    # End of round1, reveal public card (chance node) and proceed to round2
                    util0 = util1 = 0.0
                    remaining_cards = [card for card in range(DECK_SIZE) if card not in (p0_card, p1_card)]
                    for pub in remaining_cards:
                        prob = 1.0 / len(remaining_cards)
                        # Include chance probability in reach for both players
                        new_reach0 = reach0 * prob
                        new_reach1 = reach1 * prob
                        result = cfr(0, 2, p0_card, p1_card, pub,
                                     p0_contrib, p1_contrib, to_call=0, raise_count=0,
                                     history=history + "|", reach0=new_reach0, reach1=new_reach1)
                        util0 += prob * result[0]
                        util1 += prob * result[1]
                    branch_util = (util0, util1)
                else:
                    # End of round2, go to showdown (round_index becomes 3)
                    branch_util = cfr(None, 3, p0_card, p1_card, public_card,
                                      p0_contrib, p1_contrib, to_call=0, raise_count=0,
                                      history=history + "c", reach0=reach0, reach1=reach1)
            elif action == "ch":  # Check
                if to_call != 0:
                    raise ValueError("Check action with to_call > 0")
                if current_player == 1:
                    # Player 1 checks back -> end of current round
                    if round_index == 1:
                        # Both checked in round1
                        util0 = util1 = 0.0
                        remaining_cards = [card for card in range(DECK_SIZE) if card not in (p0_card, p1_card)]
                        for pub in remaining_cards:
                            prob = 1.0 / len(remaining_cards)
                            new_reach0 = reach0 * prob
                            new_reach1 = reach1 * prob
                            result = cfr(0, 2, p0_card, p1_card, pub,
                                         p0_contrib, p1_contrib, to_call=0, raise_count=0,
                                         history=history + "c|", reach0=new_reach0, reach1=new_reach1)
                            util0 += prob * result[0]
                            util1 += prob * result[1]
                        branch_util = (util0, util1)
                    else:
                        # Both checked in round2 -> showdown
                        branch_util = cfr(None, 3, p0_card, p1_card, public_card,
                                          p0_contrib, p1_contrib, to_call=0, raise_count=0,
                                          history=history + "c", reach0=reach0, reach1=reach1)
                else:
                    # Player 0 checks -> pass turn to player 1 in the same round
                    branch_util = cfr(1, round_index, p0_card, p1_card, public_card,
                                      p0_contrib, p1_contrib, to_call=0, raise_count=raise_count,
                                      history=history + "c", reach0=reach0, reach1=reach1)
            elif action == "b":  # Bet
                if to_call != 0:
                    raise ValueError("Bet action with to_call > 0")
                bet_amount = BET_SIZE_ROUND1 if round_index == 1 else BET_SIZE_ROUND2
                if current_player == 0:
                    p0_contrib += bet_amount
                else:
                    p1_contrib += bet_amount
                # Set the amount the opponent must call
                new_to_call = bet_amount
                # Opponent's turn
                branch_util = cfr(1 - current_player, round_index, p0_card, p1_card, public_card,
                                  p0_contrib, p1_contrib, to_call=new_to_call, raise_count=raise_count,
                                  history=history + "b", reach0=reach0, reach1=reach1)
            elif action == "r":  # Raise
                if to_call == 0 or raise_count >= MAX_RAISES_PER_ROUND:
                    raise ValueError("Raise action not allowed")
                raise_amount = BET_SIZE_ROUND1 if round_index == 1 else BET_SIZE_ROUND2
                # Call the outstanding bet and then raise
                if current_player == 0:
                    p0_contrib += to_call + raise_amount
                else:
                    p1_contrib += to_call + raise_amount
                # Opponent now faces a new bet of 'raise_amount'
                new_to_call = raise_amount
                # Opponent's turn after the raise
                branch_util = cfr(1 - current_player, round_index, p0_card, p1_card, public_card,
                                  p0_contrib, p1_contrib, to_call=new_to_call, raise_count=raise_count + 1,
                                  history=history + "r", reach0=reach0, reach1=reach1)
            else:
                branch_util = (0.0, 0.0)
            # Restore state variables
            p0_contrib, p1_contrib = prev_p0_contrib, prev_p1_contrib
            to_call, raise_count = prev_to_call, prev_raise_count
            history, public_card, round_index = prev_history, prev_public, prev_round
            # Accumulate utilities
            action_utils.append(branch_util)
            node_util0 += strategy[idx] * branch_util[0]
            node_util1 += strategy[idx] * branch_util[1]
        # Update regrets for the player acting, if this player is being updated this iteration
        if current_player == update_player:
            for idx, action in enumerate(actions):
                if current_player == 0:
                    # Regret for player 0
                    action_util = action_utils[idx][0]
                    node_util = node_util0
                    opp_reach = reach1
                else:
                    # Regret for player 1
                    action_util = action_utils[idx][1]
                    node_util = node_util1
                    opp_reach = reach0
                regret = action_util - node_util
                regrets[info_str][idx] += opp_reach * regret
        # Return the expected utility of this node (for both players)
        return (node_util0, node_util1)
    else:
        # Chance nodes (card dealing) are handled by explicit loops in the traversal
        return (0.0, 0.0)

# Train CFR for the specified number of iterations
iterations = 100000
start_time = time.time()
for t in range(1, iterations + 1):
    current_iter = t
    # Alternate which player updates regrets this iteration
    update_player = 0 if (t % 2 == 0) else 1
    # Full traversal of chance outcomes: loop over all possible private card deals
    for card0 in range(DECK_SIZE):
        for card1 in range(DECK_SIZE):
            if card1 == card0:
                continue
            # Both players ante 1 chip
            p0_contrib = ANTE
            p1_contrib = ANTE
            # Initial chance probability for this deal (not used in decision logic, but we include it in reach)
            deal_prob = 1.0 / (DECK_SIZE * (DECK_SIZE - 1))
            # Start recursion at player 0's decision in round1
            cfr(0, 1, card0, card1, public_card=None,
                p0_contrib=p0_contrib, p1_contrib=p1_contrib,
                to_call=0, raise_count=0, history="",
                reach0=deal_prob, reach1=deal_prob)
end_time = time.time()

# Compute the average strategy from accumulated strategy_sum
average_strategy = {}
for info, sum_probs in strategy_sum.items():
    total = sum(sum_probs)
    if total > 1e-9:
        avg_strategy = [p / total for p in sum_probs]
    else:
        # If never visited, default to uniform
        avg_strategy = [1.0 / len(sum_probs)] * len(sum_probs)
    average_strategy[info] = avg_strategy

# Evaluate the average strategy by computing the expected value for player 0
total_util_p0 = 0.0
for card0 in range(DECK_SIZE):
    for card1 in range(DECK_SIZE):
        if card1 == card0:
            continue
        # Chance probability of this deal
        deal_prob = 1.0 / (DECK_SIZE * (DECK_SIZE - 1))
        # Both players ante 1
        p0_contrib = ANTE
        p1_contrib = ANTE
        # Function to simulate the game using the fixed average strategy profile
        def simulate(player, round_index, p0_card, p1_card, public_card,
                     p0_contrib, p1_contrib, to_call, raise_count, history):
            if round_index > 2:
                # Showdown outcome from player 0's perspective
                pub_rank = public_card // CARDS_PER_RANK
                p0_rank = p0_card // CARDS_PER_RANK
                p1_rank = p1_card // CARDS_PER_RANK
                p0_pair = (p0_rank == pub_rank)
                p1_pair = (p1_rank == pub_rank)
                if p0_pair and not p1_pair:
                    return p1_contrib   # Player 0 wins opponent's contribution
                elif p1_pair and not p0_pair:
                    return -p0_contrib  # Player 0 loses their contribution
                else:
                    if p0_rank > p1_rank:
                        return p1_contrib
                    elif p1_rank > p0_rank:
                        return -p0_contrib
                    else:
                        return 0.0
            if player is None:
                return 0.0
            # Determine available actions
            if to_call > 0:
                actions = ["f", "c"]
                if raise_count < MAX_RAISES_PER_ROUND:
                    actions.append("r")
            else:
                actions = ["ch", "b"]
            info_str = info_key(player, p0_card if player == 0 else p1_card, public_card, history)
            strategy = average_strategy.get(info_str, [1.0 / len(actions)] * len(actions))
            # Expected utility from this node for player 0 under avg strategy
            util = 0.0
            for idx, action in enumerate(actions):
                prob = strategy[idx]
                if action == "f":
                    util_branch = (-p0_contrib) if player == 0 else (p1_contrib)
                elif action == "c":
                    new_p0_contrib, new_p1_contrib = p0_contrib, p1_contrib
                    if player == 0:
                        new_p0_contrib += to_call
                    else:
                        new_p1_contrib += to_call
                    if round_index == 1:
                        # Move to round2 after revealing public card
                        branch_util = 0.0
                        remaining_cards = [card for card in range(DECK_SIZE) if card not in (p0_card, p1_card)]
                        for pub in remaining_cards:
                            prob_pub = 1.0 / len(remaining_cards)
                            branch_util += prob_pub * simulate(0, 2, p0_card, p1_card, pub,
                                                               new_p0_contrib, new_p1_contrib,
                                                               to_call=0, raise_count=0, history=history + "|")
                        util_branch = branch_util
                    else:
                        # End of round2 -> showdown
                        util_branch = simulate(None, 3, p0_card, p1_card, public_card,
                                               new_p0_contrib, new_p1_contrib, to_call=0, raise_count=0,
                                               history=history + "c")
                elif action == "ch":
                    if player == 1:
                        # P1 checking ends the round
                        if round_index == 1:
                            branch_util = 0.0
                            remaining_cards = [card for card in range(DECK_SIZE) if card not in (p0_card, p1_card)]
                            for pub in remaining_cards:
                                prob_pub = 1.0 / len(remaining_cards)
                                branch_util += prob_pub * simulate(0, 2, p0_card, p1_card, pub,
                                                                   p0_contrib, p1_contrib,
                                                                   to_call=0, raise_count=0, history=history + "c|")
                            util_branch = branch_util
                        else:
                            util_branch = simulate(None, 3, p0_card, p1_card, public_card,
                                                   p0_contrib, p1_contrib, to_call=0, raise_count=0,
                                                   history=history + "c")
                    else:
                        # P0 checks -> P1's turn
                        util_branch = simulate(1, round_index, p0_card, p1_card, public_card,
                                               p0_contrib, p1_contrib, to_call=0, raise_count=raise_count,
                                               history=history + "c")
                elif action == "b":
                    bet_amount = BET_SIZE_ROUND1 if round_index == 1 else BET_SIZE_ROUND2
                    new_p0_contrib, new_p1_contrib = p0_contrib, p1_contrib
                    if player == 0:
                        new_p0_contrib += bet_amount
                    else:
                        new_p1_contrib += bet_amount
                    util_branch = simulate(1 - player, round_index, p0_card, p1_card, public_card,
                                            new_p0_contrib, new_p1_contrib,
                                            to_call=bet_amount, raise_count=raise_count, history=history + "b")
                elif action == "r":
                    raise_amount = BET_SIZE_ROUND1 if round_index == 1 else BET_SIZE_ROUND2
                    new_p0_contrib, new_p1_contrib = p0_contrib, p1_contrib
                    if player == 0:
                        new_p0_contrib += to_call + raise_amount
                    else:
                        new_p1_contrib += to_call + raise_amount
                    util_branch = simulate(1 - player, round_index, p0_card, p1_card, public_card,
                                            new_p0_contrib, new_p1_contrib,
                                            to_call=raise_amount, raise_count=raise_count + 1, history=history + "r")
                else:
                    util_branch = 0.0
                util += prob * util_branch
            return util
        # Simulate the game for this deal using the average strategy profile
        util_p0 = simulate(0, 1, card0, card1, public_card=None,
                           p0_contrib=p0_contrib, p1_contrib=p1_contrib,
                           to_call=0, raise_count=0, history="")
        total_util_p0 += deal_prob * util_p0

# First player (player 0) average utility under the average strategy
avg_utility_p0 = total_util_p0
elapsed_time = end_time - start_time
print(f"Player 0 average utility after {iterations} iterations: {avg_utility_p0:.4f}")
print(f"Elapsed time: {elapsed_time:.2f} seconds")


