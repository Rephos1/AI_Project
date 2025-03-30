import numpy as np

class RPSTrainer:
    ROCK, PAPER, SCISSORS = 0, 1, 2
    NUM_ACTIONS = 3
    
    def __init__(self):
        self.regret_sum = np.zeros(self.NUM_ACTIONS)
        self.strategy_sum = np.zeros(self.NUM_ACTIONS)
        # Example fixed opponent strategy 
        self.opponent_strategy = np.array([0.3, 0.3, 0.4])
        
        # Payoff matrix for [my_action][opponent_action]
        self.payoff_matrix = np.array([
            #R   #P   #S
            [ 0, -1,  1],  # Rock
            [ 1,  0, -1],  # Paper
            [-1,  1,  0]   # Scissors
        ])
    
    def get_strategy(self):
        """Calculate current strategy through regret-matching."""
        strategy = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(strategy)
        
        if normalizing_sum > 0:
            strategy = strategy / normalizing_sum
        else:
            strategy = np.ones(self.NUM_ACTIONS) / self.NUM_ACTIONS
            
        self.strategy_sum += strategy
        return strategy
    
    def get_action(self, strategy):
        """Sample an action from the strategy probability distribution."""
        return np.random.choice(self.NUM_ACTIONS, p=strategy)
    
    def get_average_strategy(self):
        """Get the average strategy across all training iterations."""
        strategy_sum = np.sum(self.strategy_sum)
        if strategy_sum > 0:
            return self.strategy_sum / strategy_sum
        return np.ones(self.NUM_ACTIONS) / self.NUM_ACTIONS
    
    def train(self, iterations=1000):
        """Train the CFR algorithm for specified iterations."""
        for _ in range(iterations):
            # Get current strategy and sample actions
            strategy = self.get_strategy()
            my_action = self.get_action(strategy)
            opponent_action = self.get_action(self.opponent_strategy)
            
            # Calculate action utilities
            action_utilities = self.payoff_matrix[:, opponent_action]
            if _ %1000 == 0:
                print(f"\n Action utilities: {action_utilities}")
                print(f"Strategy: {strategy}")
                print(f"My action: {my_action}")
                print(f"Opponent action: {opponent_action}")
                print(f"Regret sum: {self.regret_sum}")
                print(f"Strategy sum: {self.strategy_sum}") 
            # Calculate regret
            realized_utility = action_utilities[my_action]
            regrets = action_utilities - realized_utility
            
            # Accumulate regret
            self.regret_sum += regrets

def main():
    trainer = RPSTrainer()
    trainer.train(10000)
    
    avg_strategy = trainer.get_average_strategy()
    print("\nFinal strategy probabilities:")
    print(f"Rock: {avg_strategy[0]:.3f}")
    print(f"Paper: {avg_strategy[1]:.3f}")
    print(f"Scissors: {avg_strategy[2]:.3f}")
    print(f"Average strategy: {avg_strategy[0]}")
if __name__ == "__main__":
    main()
