import random
import numpy as np

class RPSTrainer:
    ROCK, PAPER, SCISSORS = 0, 1, 2
    NUM_ACTIONS = 3
    
    def __init__(self):
        self.regret_sum = np.zeros(self.NUM_ACTIONS)
        self.strategy_sum = np.zeros(self.NUM_ACTIONS)
        
        # Payoff matrix for [my_action][opponent_action]
        self.payoff_matrix = np.array([
            [ 0, -1,  1],  # Rock
            [ 1,  0, -1],  # Paper
            [-1,  1,  0]   # Scissors
        ])
    
    def get_strategy(self):
        strategy = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(strategy)
        
        if normalizing_sum > 0:
            strategy = strategy / normalizing_sum
        else:
            strategy = np.ones(self.NUM_ACTIONS) / self.NUM_ACTIONS
            
        self.strategy_sum += strategy
        return strategy
    
    def get_action(self, strategy):
        return np.random.choice(self.NUM_ACTIONS, p=strategy)
    
    def get_average_strategy(self):
        strategy_sum = np.sum(self.strategy_sum)
        if strategy_sum > 0:
            return self.strategy_sum / strategy_sum
        return np.ones(self.NUM_ACTIONS) / self.NUM_ACTIONS
    
    def update_regret(self, my_action, opponent_action, strategy):
        """Update regrets based on actions played"""
        action_utilities = self.payoff_matrix[:, opponent_action]
        realized_utility = action_utilities[my_action]
        regrets = action_utilities - realized_utility
        self.regret_sum += regrets

def train_two_players(iterations=10000):
    """Train two players simultaneously"""
    player1 = RPSTrainer()
    player2 = RPSTrainer()
    
    for _ in range(iterations):
        # Get current strategies
        strategy1 = player1.get_strategy()
        strategy2 = player2.get_strategy()
        
        # Sample actions
        action1 = player1.get_action(strategy1)
        action2 = player2.get_action(strategy2)
        
        # Update regrets for both players
        player1.update_regret(action1, action2, strategy1)
        # Note: Player 2's payoff matrix is the negative of player 1's
        player2.update_regret(action2, action1, strategy2)
    
    return player1, player2

def main():
    print("Training two players against each other...")
    player1, player2 = train_two_players(100000)
    
    p1_strategy = player1.get_average_strategy()
    p2_strategy = player2.get_average_strategy()
    
    print("\nPlayer 1 final strategy:")
    print(f"Rock: {p1_strategy[0]:.3f}")
    print(f"Paper: {p1_strategy[1]:.3f}")
    print(f"Scissors: {p1_strategy[2]:.3f}")
    
    print("\nPlayer 2 final strategy:")
    print(f"Rock: {p2_strategy[0]:.3f}")
    print(f"Paper: {p2_strategy[1]:.3f}")
    print(f"Scissors: {p2_strategy[2]:.3f}")
    
if __name__ == "__main__":
    main()
