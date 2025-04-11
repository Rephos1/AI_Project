import random
import copy
##############################
# Card and Game State Classes
##############################

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit  # suit is a string such as "H" or "S"

    def __repr__(self):
        return f"{self.rank}{self.suit}"


class LeducGameState:
    """
    Represents the state of a Leduc Hold’Em game.
    
    Attributes:
      - round: 1 for the preflop (private cards) round, 2 for the board round.
      - deck: list of remaining Card objects.
      - private_cards: list of two cards (one per player).
      - board: the single community card (set in round 2).
      - pot: current size of the pot.
      - contributions: chips put in the current round (used to compute required call amounts).
      - total_contributions: total chips contributed by each player in the game.
      - bet_count: number of aggressive actions (bet/raise) taken in the current round.
      - current_player: the index (0 or 1) of the player whose turn it is.
      - last_action: list of recent actions in the current betting round.
      - terminal: Boolean flag indicating if the game is over.
      - winner: If the game ended by a fold, winner holds the winning player’s index.
    """
    def __init__(self):
        self.round = 1  # 1: preflop, 2: board round
        self.deck = []
        self.private_cards = [None, None]  # private cards for players 0 and 1
        self.board = None                # board card for round 2
        self.pot = 2                     # initial pot (each player posts an ante of 1)
        self.contributions = [1, 1]        # contributions in the current round (start with ante)
        self.total_contributions = [1, 1]  # overall contributions per player in the game
        self.bet_count = 0               # number of bet/raise actions in the current betting round
        self.current_player = 0          # in round 1, player 0 acts first
        self.last_action = []            # records the actions taken in the current round
        self.terminal = False            # flag for end-of-game
        self.winner = None             # if game ends by a fold, set the winning player's index


##############################
# Leduc Hold'Em Game Class
##############################

class LeducHoldemGame:
    """
    Implements Leduc Hold’Em game logic with two betting actions per round.
    
    Game flow:
      1. Game start:
          - Create and shuffle a deck of 6 cards (2 suits × 3 ranks).
          - Deal one private card to each player.
          - Each player posts an ante of 1 (pot=2).
      2. Preflop Betting Round (round==1):
          - Legal actions (when no bet is outstanding): "check" and "bet" (bet adds 2 chips).
          - If a bet is made, opponent can "call", "fold", or (if not already raised) "raise".
          - A raise adds the amount needed to call plus 2 extra chips.
          - Two bets max in the round.
      3. Transition to Second Round:
          - If no one folds, deal one board card.
          - Reset the current round contributions and bet count.
          - In round 2, the bet (or raise) amounts are 4 chips.
      4. Second Betting Round (round==2):
          - Same structure as round 1 but with bet/raise amount of 4.
      5. Terminal Node / Showdown:
          - If a player folds, the other wins immediately.
          - Otherwise, evaluate both players’ hands:
              * A pair (private card matching the board) beats a high–card hand.
              * When both hands are of the same type, the card rank (J < Q < K) decides.
              * Ties split the pot.
          - The payoff for each player is computed as (winning chips minus that player’s total contributions),
            so that payoffs are zero–sum.
    """

    def __init__(self):
        self.ranks = ['J', 'Q', 'K']
        self.suits = ['H', 'S']  # e.g. Hearts and Spades
        # Bet amounts are fixed per round: 2 chips for round 1 and 4 chips for round 2.
        self.bet_amounts = {1: 2, 2: 4}
        self.max_bets = 2  # Maximum aggressive actions (bet + raise) allowed each round.
        self.state = None
        self.reset_game()

    def reset_game(self):
        """Starts a new game by shuffling the deck, dealing private cards, and initializing state."""
        # Build deck: two copies of each rank in different suits.
        deck = [Card(rank, suit) for rank in self.ranks for suit in self.suits]
        random.shuffle(deck)
        state = LeducGameState()
        state.deck = deck.copy()
        # Deal one private card to each player.
        state.private_cards[0] = state.deck.pop()
        state.private_cards[1] = state.deck.pop()
        self.state = state

    def get_legal_actions(self):
        """
        Returns the list of legal actions for the current decision node.
        
        When in a chance node for dealing the board, only the 'deal_board' action is legal.
        Otherwise, if no bet is outstanding, the actions are "check" and "bet".
        When facing a bet, the legal actions are "fold" and "call" (and "raise" if no raise has been made yet).
        """
        s = self.state
        if s.terminal:
            return []
        # Chance node: if in round 2 and the board has not been dealt.
        if s.round == 2 and s.board is None:
            return ['deal_board']

        # Decision node:
        current_contribution = s.contributions[s.current_player]
        opponent = 1 - s.current_player
        required_call = s.contributions[opponent] - current_contribution

        if required_call == 0:
            # No outstanding bet: "check" or "bet" (placing the first bet).
            return ['check', 'bet']
        else:
            # There is an outstanding bet; options are "fold" or "call".
            # Additionally, if no raise has been taken (i.e. bet_count < max_bets) a "raise" is allowed.
            actions = ['fold', 'call']
            if s.bet_count < self.max_bets:
                actions.append('raise')
            return actions

    def apply_action(self, action):
        """
        Advances the game state by applying the given action.
        
        Supported actions:
          - "deal_board": Used at the beginning of round 2 to deal the board card.
          - "check": When no bet is outstanding. If both players check consecutively, the round ends.
          - "bet": Place a bet (amount depends on the round).
          - "call": Match the outstanding bet and then end the betting round.
          - "raise": First call the outstanding bet, then add the raise amount.
          - "fold": End the game with the current player folding.
        """
        s = self.state
        if s.terminal:
            return

        # Chance node: deal board card.
        if s.round == 2 and s.board is None:
            if action == 'deal_board':
                s.board = s.deck.pop()
                # Prepare for round 2 betting.
                s.contributions = [0, 0]
                s.bet_count = 0
                s.last_action = []
                # In round 2, the typical order is that player 1 acts first.
                s.current_player = 1
            else:
                raise ValueError("Invalid action at chance node: expected 'deal_board'")
            return

        # Decision actions.
        if action == 'check':
            # When checking with no bet outstanding, if the previous action was a check then end the round.
            if s.last_action and s.last_action[-1] == 'check':
                self.end_round()
            else:
                s.last_action.append('check')
                s.current_player = 1 - s.current_player

        elif action == 'bet':
            # Player bets when no bet is outstanding.
            bet = self.bet_amounts[s.round]
            s.contributions[s.current_player] += bet
            s.total_contributions[s.current_player] += bet
            s.pot += bet
            s.bet_count = 1  # first bet this round
            s.last_action.append('bet')
            s.current_player = 1 - s.current_player

        elif action == 'call':
            # Match the outstanding bet and then end the round.
            opponent = 1 - s.current_player
            required_call = s.contributions[opponent] - s.contributions[s.current_player]
            s.contributions[s.current_player] += required_call
            s.total_contributions[s.current_player] += required_call
            s.pot += required_call
            self.end_round()

        elif action == 'raise':
            # To raise, the acting player must first call the outstanding bet, then add the raise amount.
            opponent = 1 - s.current_player
            required_call = s.contributions[opponent] - s.contributions[s.current_player]
            raise_amount = self.bet_amounts[s.round]
            total_raise = required_call + raise_amount
            s.contributions[s.current_player] += total_raise
            s.total_contributions[s.current_player] += total_raise
            s.pot += total_raise
            s.bet_count += 1  # now two aggressive actions have occurred this round
            s.last_action.append('raise')
            s.current_player = 1 - s.current_player

        elif action == 'fold':
            # Folding ends the game immediately. The opponent is declared the winner.
            s.terminal = True
            s.winner = 1 - s.current_player

        else:
            raise ValueError(f"Unknown action: {action}")

    def end_round(self):
        """
        Ends the current betting round.
        
        In round 1 (preflop), the game moves to round 2 by dealing the board card and resetting
        contributions and betting count. In round 2, ending the round causes the game to reach a terminal
        state (going to showdown).
        """
        s = self.state
        if s.round == 1:
            # Transition from preflop to second round.
            s.round = 2
            s.last_action = []
            # Deal the board card as a chance event.
            s.board = s.deck.pop()
            # Reset round contributions and bet count.
            s.contributions = [0, 0]
            s.bet_count = 0
            # Typically in round 2, player 1 acts first.
            s.current_player = 1
        else:
            # Round 2: finish the game (showdown).
            s.terminal = True

    def is_terminal(self):
        """Returns True if the game has reached a terminal state."""
        return self.state.terminal

    def showdown(self):
        """
        Computes the final payoff for each player when the game is over.
        
        Payoffs are computed as:
          payoff = (winnings from the pot) - (total chips contributed)
          
        When the game ends by fold, the non–folding player wins the entire pot.
        At showdown, each player's hand is evaluated as follows:
          - If a player's private card matches the board card, the player has a pair.
          - A pair beats a high–card hand.
          - In case both hands are of the same type, the numeric rank is compared
            (using the ordering J < Q < K).
          - Ties split the pot.
          
        Returns:
          A list of two payoffs [payoff_player0, payoff_player1].
        """
        s = self.state
        if not s.terminal:
            raise ValueError("Game is not terminal")
        
        # If the game ended by a fold.
        if s.winner is not None:
            payoffs = [0, 0]
            winner = s.winner
            payoffs[winner] = s.pot - s.total_contributions[winner]
            loser = 1 - winner
            payoffs[loser] = -s.total_contributions[loser]
            return payoffs

        # At showdown, evaluate hands.
        rank_value = {'J': 0, 'Q': 1, 'K': 2}

        def hand_value(card, board):
            """Return a tuple (is_pair, rank) where is_pair is 1 if card matches board, else 0."""
            if board is not None and card.rank == board.rank:
                return (1, rank_value[card.rank])
            else:
                return (0, rank_value[card.rank])

        hv0 = hand_value(s.private_cards[0], s.board)
        hv1 = hand_value(s.private_cards[1], s.board)

        if hv0 > hv1:
            winner = 0
        elif hv1 > hv0:
            winner = 1
        else:
            winner = None  # tie

        payoffs = [0, 0]
        if winner is None:
            # Split the pot equally.
            payoffs[0] = s.pot / 2 - s.total_contributions[0]
            payoffs[1] = s.pot / 2 - s.total_contributions[1]
        else:
            payoffs[winner] = s.pot - s.total_contributions[winner]
            loser = 1 - winner
            payoffs[loser] = -s.total_contributions[loser]
        return payoffs

    def get_state(self):
        """Returns the current game state (useful for debugging or CFR information set extraction)."""
        return self.state


