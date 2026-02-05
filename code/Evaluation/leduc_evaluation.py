#!/usr/bin/env python3
# evaluation/evaluate_leduc.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path to import the LeducCFR class
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the LeducCFR class from the parent directory
from leduc_holdem import LeducCFR, Node, ACTIONS

class LeducEvaluator:
    """
    Class for evaluating Leduc Poker CFR implementation with various metrics
    and visualization tools.
    """
    
    def __init__(self):
        self.base_iterations = 1000
        self.iteration_multiples = [1, 5, 10, 50, 100]  # Different scale factors
        self.exploitability_data = []
        self.convergence_data = []
        self.strategy_evolution = defaultdict(list)
        self.runtime_data = []
    
    def measure_exploitability(self, leduc, iterations):
        """
        Estimate exploitability of the current strategy.
        Lower exploitability indicates closer convergence to Nash equilibrium.
        """
        # A simple exploitability measure: how much a best response player can gain
        # against the current strategy
        exploitability = 0
        
        # Sample some key information sets
        sample_cards = [(0, 1), (2, 3), (4, 5)]  # Some representative card combinations
        
        for p0_card, p1_card in sample_cards:
            # Test against player 0's strategy
            best_response_value = self._get_best_response_value(leduc, p0_card, p1_card, 0)
            exploitability += best_response_value
            
            # Test against player 1's strategy  
            best_response_value = self._get_best_response_value(leduc, p0_card, p1_card, 1)
            exploitability += best_response_value
        
        # Normalize by number of samples
        return exploitability / (len(sample_cards) * 2)
    
    def _get_best_response_value(self, leduc, p0_card, p1_card, player_idx):
        """Compute best response value for a specific card combination against a player"""
        # Simplified best response computation for demonstration purposes
        pot = 2 * leduc.ante
        history = ""
        
        # Force deterministic cards for this evaluation
        leduc.deck = [p0_card, p1_card] + [c for c in range(6) if c != p0_card and c != p1_card]
        
        # Get the value of best response for different starting actions
        max_value = float('-inf')
        for action in ACTIONS:
            next_history = action
            next_pot = pot
            if action == 'b':
                next_pot += leduc.bet_r1
                
            # Follow the optimal strategy for the opponent
            # NOTE: This is a simplified version; a true best response would require
            # recursively computing the best action at every decision point
            if player_idx == 0:
                value = self._simulate_play(leduc, p0_card, p1_card, next_history, next_pot, fixed_strategy_player=1)
            else:
                value = -self._simulate_play(leduc, p0_card, p1_card, next_history, next_pot, fixed_strategy_player=0)
            
            max_value = max(max_value, value)
            
        return max_value
    
    def _simulate_play(self, leduc, p0_card, p1_card, history, pot, fixed_strategy_player, depth=0):
        """
        Simulate play with one player following the learned strategy and the other exploring.
        Returns the expected value from the first player's perspective.
        """
        # Limit recursion depth for this demonstration
        if depth > 6:  
            return 0
            
        plays = len(history)
        acting = plays % 2
        
        # Terminal conditions
        if (plays >= 2 and "r1" not in history) or history.endswith('f'):
            # Simple payoff estimation
            if history.endswith('f'):
                return pot if (len(history) - 1) % 2 != 0 else -pot
            return pot/2  # Simplified estimate
            
        # Acting player follows strategy if they're the fixed strategy player
        if acting == fixed_strategy_player:
            if acting == 0:
                infoset = f"R1|{p0_card}|{history}"
            else:
                infoset = f"R1|{p1_card}|{history}"
                
            # Get the strategy for this infoset
            if infoset in leduc.node_map:
                node = leduc.node_map[infoset]
                strategy = node.get_average_strategy()
                
                # Compute expected value of following this strategy
                value = 0
                for a in range(len(ACTIONS)):
                    next_history = history + ACTIONS[a]
                    next_pot = pot
                    if ACTIONS[a] == 'b':
                        next_pot += leduc.bet_r1
                    
                    action_value = self._simulate_play(
                        leduc, p0_card, p1_card, next_history, next_pot, 
                        fixed_strategy_player, depth+1
                    )
                    value += strategy[a] * action_value
                return value
            else:
                # No strategy yet, use uniform
                return 0
        else:
            # For the exploring player, try each action
            best_value = float('-inf')
            for a in range(len(ACTIONS)):
                next_history = history + ACTIONS[a]
                next_pot = pot
                if ACTIONS[a] == 'b':  
                    next_pot += leduc.bet_r1
                
                action_value = self._simulate_play(
                    leduc, p0_card, p1_card, next_history, next_pot, 
                    fixed_strategy_player, depth+1
                )
                
                # If we're player 1, we need to negate the values
                if acting == 1:
                    action_value = -action_value
                    
                best_value = max(best_value, action_value)
            
            return best_value if acting == 0 else -best_value
    
    def track_strategy_evolution(self, leduc):
        """Track how strategies evolve for select information sets"""
        # Choose some interesting information sets to track
        key_infosets = [
            "R1|0|",       # Player 0, round 1, holding lowest card, starting action
            "R1|4|",       # Player 0, round 1, holding highest card, starting action
            "R1|2|b",      # Player 1, round 1, middle card, responding to bet
            "R2|0|2|",     # Player 0, round 2, low card with middle board
            "R2|4|0|b"     # Player 1, round 2, high card, low board, responding to bet
        ]
        
        strategies = {}
        for infoset in key_infosets:
            if infoset in leduc.node_map:
                node = leduc.node_map[infoset]
                strategies[infoset] = node.get_average_strategy().copy()
            else:
                strategies[infoset] = np.array([1/3, 1/3, 1/3])  # Uniform if not found
                
        return strategies
    
    def run_evaluation(self):
        """Run complete evaluation with increasing iterations"""
        print("Starting Leduc Poker CFR Evaluation...")
        
        for multiple in self.iteration_multiples:
            iterations = self.base_iterations * multiple
            print(f"\nRunning evaluation with {iterations} iterations...")
            
            # Run CFR
            start_time = time.time()
            leduc = LeducCFR(iterations)
            avg_util = leduc.cfr()
            runtime = time.time() - start_time
            
            # Gather data
            exploitability = self.measure_exploitability(leduc, iterations)
            self.exploitability_data.append((iterations, exploitability))
            self.convergence_data.append((iterations, avg_util[0]))  # Player 0's perspective
            self.runtime_data.append((iterations, runtime))
            
            # Track strategy evolution
            strategies = self.track_strategy_evolution(leduc)
            for infoset, strategy in strategies.items():
                self.strategy_evolution[infoset].append((iterations, strategy))
            
            print(f"Completed {iterations} iterations in {runtime:.2f} seconds")
            print(f"Exploitability: {exploitability:.6f}")
            print(f"Average utility: {avg_util}")
        
        # Generate all plots
        self.generate_plots()
        
    def generate_plots(self):
        """Generate all evaluation plots"""
        self._plot_exploitability()
        self._plot_convergence()
        self._plot_runtime()
        self._plot_strategy_evolution()
        self._plot_combined_metrics()
        
    def _plot_exploitability(self):
        """Plot exploitability vs iterations"""
        plt.figure(figsize=(10, 6))
        x, y = zip(*self.exploitability_data)
        plt.plot(x, y, 'o-', linewidth=2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('CFR Iterations')
        plt.ylabel('Exploitability')
        plt.title('Exploitability vs. Iterations in Leduc Poker')
        plt.grid(True, which="both", ls="--")
        plt.savefig('exploitability.png', dpi=300, bbox_inches='tight')
        
    def _plot_convergence(self):
        """Plot convergence of average utility"""
        plt.figure(figsize=(10, 6))
        x, y = zip(*self.convergence_data)
        plt.plot(x, y, 'o-', linewidth=2)
        plt.xscale('log')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('CFR Iterations')
        plt.ylabel('Average Utility (Player 0)')
        plt.title('Convergence of Average Utility in Leduc Poker')
        plt.grid(True, which="both", ls="--")
        plt.savefig('convergence.png', dpi=300, bbox_inches='tight')
        
    def _plot_runtime(self):
        """Plot runtime vs iterations"""
        plt.figure(figsize=(10, 6))
        x, y = zip(*self.runtime_data)
        plt.plot(x, y, 'o-', linewidth=2)
        plt.xscale('log')
        plt.xlabel('CFR Iterations')
        plt.ylabel('Runtime (seconds)')
        plt.title('Runtime vs. Iterations in Leduc Poker')
        plt.grid(True, which="both", ls="--")
        plt.savefig('runtime.png', dpi=300, bbox_inches='tight')
        
    def _plot_strategy_evolution(self):
        """Plot how strategies evolve for select information sets"""
        # Plot strategy evolution for each tracked infoset
        for infoset, data in self.strategy_evolution.items():
            plt.figure(figsize=(10, 6))
            
            # Extract data
            iterations = [point[0] for point in data]
            strategies = [point[1] for point in data]
            
            # Stack strategies for each action
            check_probs = [s[0] for s in strategies]
            bet_probs = [s[1] for s in strategies]
            fold_probs = [s[2] for s in strategies]
            
            plt.plot(iterations, check_probs, 'o-', label='Check/Call')
            plt.plot(iterations, bet_probs, 's-', label='Bet/Raise')
            plt.plot(iterations, fold_probs, '^-', label='Fold')
            
            plt.xscale('log')
            plt.xlabel('CFR Iterations')
            plt.ylabel('Action Probability')
            plt.title(f'Strategy Evolution for InfoSet: {infoset}')
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.ylim(-0.05, 1.05)
            
            # Create a clean filename
            filename = f'strategy_evolution_{infoset.replace("|", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
    def _plot_combined_metrics(self):
        """Plot combined metrics on log-log scale"""
        # Create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Plot exploitability
        x1, y1 = zip(*self.exploitability_data)
        line1 = ax1.plot(x1, y1, 'o-', color='blue', linewidth=2, label='Exploitability')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('CFR Iterations')
        ax1.set_ylabel('Exploitability', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create second y-axis
        ax2 = ax1.twinx()
        x2, y2 = zip(*self.runtime_data)
        line2 = ax2.plot(x2, y2, 's-', color='red', linewidth=2, label='Runtime (seconds)')
        ax2.set_yscale('log')
        ax2.set_ylabel('Runtime (seconds)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center')
        
        plt.title('Exploitability vs. Runtime Trade-off in Leduc Poker')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.savefig('combined_metrics.png', dpi=300, bbox_inches='tight')
        
    def analyze_strategy(self, iterations=10000):
        """
        Perform detailed analysis of final strategy for specific hands/situations.
        Returns text analysis and creates a heatmap visualization.
        """
        print("\nPerforming detailed strategy analysis...")
        leduc = LeducCFR(iterations)
        leduc.cfr()
        
        # Analysis data structures
        strategies_by_rank = defaultdict(list)
        round1_strategies = {}
        round2_strategies = {}
        
        # Collect strategy data
        for infoset, node in leduc.node_map.items():
            parts = infoset.split('|')
            if len(parts) < 3:
                continue
                
            if parts[0] == "R1":
                round_num = 1
                card = int(parts[1])
                history = parts[2]
                
                # Group by card rank (ignoring suit)
                card_rank = card // 2
                key = f"Rank {card_rank}, History: {history}"
                round1_strategies[key] = node.get_average_strategy()
                
            elif parts[0] == "R2" and len(parts) >= 4:
                round_num = 2
                card = int(parts[1])
                board = int(parts[2])
                history = parts[3]
                
                # Check for pairs
                is_pair = (card // 2) == (board // 2)
                key = f"{'Pair' if is_pair else 'No Pair'}, Card: {card}, Board: {board}, History: {history}"
                round2_strategies[key] = node.get_average_strategy()
        
        # Generate heatmap for round 1 strategies
        self._generate_strategy_heatmap(round1_strategies, "round1_strategy_heatmap.png", "Round 1")
        
        # Generate heatmap for round 2 strategies
        self._generate_strategy_heatmap(round2_strategies, "round2_strategy_heatmap.png", "Round 2")
        
        return round1_strategies, round2_strategies
        
    def _generate_strategy_heatmap(self, strategies, filename, title_prefix):
        """Generate strategy heatmap visualization"""
        if not strategies:
            return
            
        # Prepare data for heatmap
        situations = list(strategies.keys())
        action_labels = ["Check/Call", "Bet/Raise", "Fold"]
        
        # Create data matrix
        data = np.zeros((len(situations), len(action_labels)))
        for i, situation in enumerate(situations):
            for j, action in enumerate(action_labels):
                data[i, j] = strategies[situation][j]
        
        # Create heatmap
        plt.figure(figsize=(14, max(8, len(situations) * 0.4)))
        sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", 
                   xticklabels=action_labels, yticklabels=situations)
        plt.title(f"{title_prefix} Strategies in Leduc Poker")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    def evaluate_vs_simple_agent(self, iterations=10000, eval_games=1000):
        """Evaluate trained CFR against simple heuristic agents"""
        print("\nEvaluating against simple agents...")
        
        # Train the CFR agent
        leduc = LeducCFR(iterations)
        leduc.cfr()
        
        # Define simple agents
        agents = {
            "Always Aggressive": lambda infoset, pot: 1,  # Always bets/raises
            "Always Passive": lambda infoset, pot: 0,     # Always checks/calls
            "Random": lambda infoset, pot: np.random.choice([0, 1, 2]),
            "Simple Heuristic": self._simple_heuristic_agent
        }
        
        results = {}
        
        # Play against each agent
        for agent_name, agent_fn in agents.items():
            print(f"Evaluating against {agent_name}...")
            winnings = self._play_vs_agent(leduc, agent_fn, eval_games)
            results[agent_name] = winnings
            print(f"Average winnings vs {agent_name}: {winnings:.3f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        agents = list(results.keys())
        winnings = list(results.values())
        
        bars = plt.bar(agents, winnings)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if winnings[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.ylabel('Average Winnings per Game')
        plt.title('CFR Agent Performance vs Simple Agents')
        plt.ylim(min(min(winnings) * 1.2, -0.5), max(max(winnings) * 1.2, 0.5))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('vs_simple_agents.png', dpi=300, bbox_inches='tight')
        
        return results
    
    def _simple_heuristic_agent(self, infoset, pot):
        """A simple rule-based agent for comparison"""
        parts = infoset.split('|')
        
        if parts[0] == "R1":
            card = int(parts[1])
            history = parts[2]
            
            # High card - bet/raise
            if card >= 4:
                return 1  # Bet/raise
            # Medium card - check/call
            elif card >= 2:
                if 'b' in history:
                    return 0  # Call
                else:
                    return np.random.choice([0, 1])  # Mix of check and bet
            # Low card - mostly check/fold
            else:
                if 'b' in history:
                    return 2 if np.random.random() < 0.7 else 0  # Mostly fold, sometimes call
                else:
                    return 0  # Check
        else:  # Round 2
            card = int(parts[1])
            board = int(parts[2])
            history = parts[3]
            
            # Check for pair
            if card // 2 == board // 2:
                return 1  # Always bet with pair
            else:
                if 'b' in history:
                    return 0 if card >= 3 else 2  # Call with high card, fold with low
                else:
                    return 1 if card >= 4 else 0  # Bet with very high, check with others
    
    def _play_vs_agent(self, leduc, agent_fn, num_games):
        """Play CFR strategy against an agent"""
        total_winnings = 0
        
        for _ in range(num_games):
            # Shuffle the deck
            deck = list(range(6))
            random.shuffle(deck)
            
            # Deal cards
            p0_card = deck[0]  # CFR agent
            p1_card = deck[1]  # Simple agent
            
            # Initial pot (antes)
            pot = 2 * leduc.ante
            
            # Play round 1
            history = ""
            acting = 0
            
            while True:
                if acting == 0:  # CFR agent's turn
                    infoset = f"R1|{p0_card}|{history}"
                    if infoset in leduc.node_map:
                        node = leduc.node_map[infoset]
                        strategy = node.get_average_strategy()
                        action_idx = np.random.choice(len(ACTIONS), p=strategy)
                    else:
                        action_idx = np.random.choice(len(ACTIONS))
                else:  # Simple agent's turn
                    infoset = f"R1|{p1_card}|{history}"
                    action_idx = agent_fn(infoset, pot)
                
                action = ACTIONS[action_idx]
                history += action
                
                # Update pot
                if action == 'b':
                    pot += leduc.bet_r1
                
                # Check terminal conditions
                if action == 'f':
                    # Fold: acting player loses
                    winner = 1 - acting
                    total_winnings += pot if winner == 0 else -pot
                    break
                
                # Switch players
                acting = 1 - acting
                
                # End of round 1 check
                if len(history) >= 2:
                    # Draw board card
                    board = deck[2]
                    
                    # Start round 2
                    r2_history = ""
                    r2_acting = 0  # First player acts first in round 2
                    
                    # Play round 2
                    while True:
                        if r2_acting == 0:  # CFR agent's turn
                            infoset = f"R2|{p0_card}|{board}|{r2_history}"
                            if infoset in leduc.node_map:
                                node = leduc.node_map[infoset]
                                strategy = node.get_average_strategy()
                                action_idx = np.random.choice(len(ACTIONS), p=strategy)
                            else:
                                action_idx = np.random.choice(len(ACTIONS))
                        else:  # Simple agent's turn
                            infoset = f"R2|{p1_card}|{board}|{r2_history}"
                            action_idx = agent_fn(infoset, pot)
                        
                        action = ACTIONS[action_idx]
                        r2_history += action
                        
                        # Update pot
                        if action == 'b':
                            pot += leduc.bet_r2
                        
                        # Check terminal conditions
                        if action == 'f':
                            # Fold: acting player loses
                            winner = 1 - r2_acting
                            total_winnings += pot if winner == 0 else -pot
                            break
                        
                        # Switch players
                        r2_acting = 1 - r2_acting
                        
                        # End of round 2 check (both players acted)
                        if len(r2_history) >= 2:
                            # Showdown
                            # Determine hand strengths
                            p0_strength = 100 + p0_card if p0_card // 2 == board // 2 else p0_card
                            p1_strength = 100 + p1_card if p1_card // 2 == board // 2 else p1_card
                            
                            winner = 0 if p0_strength > p1_strength else 1
                            total_winnings += pot if winner == 0 else -pot
                            break
                    
                    # We've completed a full game
                    break
        
        # Return average winnings per game
        return total_winnings / num_games


def main():
    """Main function to run evaluation"""
    evaluator = LeducEvaluator()
    
    # Run the full evaluation
    evaluator.run_evaluation()
    
    # Analyze final strategy
    evaluator.analyze_strategy(iterations=5000)
    
    # Evaluate against simple agents
    evaluator.evaluate_vs_simple_agent(iterations=10000, eval_games=1000)
    
    print("\nEvaluation complete. All plots have been saved.")


if __name__ == "__main__":
    main()