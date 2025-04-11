import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
from tqdm import tqdm
import sys
from collections import defaultdict
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from kuhn_poker import KuhnCFR,Node
# Set the style for the plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def evaluate_convergence(iterations_list, decksize=3, runs=5):
    """
    Evaluate the convergence of KuhnCFR over different numbers of iterations.
    
    Args:
        iterations_list: List of iteration counts to evaluate
        decksize: Size of the deck to use
        runs: Number of runs for each iteration count (for statistical significance)
    
    Returns:
        DataFrame with convergence metrics
    """
    results = []
    
    for iterations in tqdm(iterations_list, desc="Testing different iteration counts"):
        for run in range(runs):
            start_time = time.time()
            
            # Initialize and run CFR
            cfr = KuhnCFR(iterations, decksize)
            cfr.cfr_iterations_external()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Calculate exploitability (approximate by checking strategy distance from Nash)
            avg_strategies = {info: node.get_average_strategy() for info, node in cfr.nodes.items()}
            
            # Count distinct information sets
            num_infosets = len(cfr.nodes)
            
            # Analyze strategy distribution
            strategy_entropy = []
            for info, node in cfr.nodes.items():
                avg_strategy = node.get_average_strategy()
                # Calculate entropy of strategy (measure of randomization)
                entropy = -np.sum(avg_strategy * np.log2(avg_strategy + 1e-10))
                strategy_entropy.append(entropy)
            
            avg_entropy = np.mean(strategy_entropy) if strategy_entropy else 0
            
            results.append({
                'iterations': iterations,
                'run': run,
                'time': elapsed_time,
                'num_infosets': num_infosets,
                'avg_entropy': avg_entropy
            })
    
    return pd.DataFrame(results)

def analyze_strategy_evolution(max_iterations=100000, decksize=3, checkpoints=10):
    """
    Analyze how strategies evolve over iterations.
    
    Args:
        max_iterations: Maximum number of iterations to run
        decksize: Size of the deck to use
        checkpoints: Number of checkpoints to record strategies at
    
    Returns:
        Dictionary mapping checkpoints to strategies
    """
    checkpoint_iterations = [int(max_iterations * i / checkpoints) for i in range(1, checkpoints + 1)]
    strategies_over_time = {}
    
    cfr = KuhnCFR(max_iterations, decksize)
    
    # Track strategies at each checkpoint
    for checkpoint_iter in tqdm(checkpoint_iterations, desc="Running strategy evolution analysis"):
        iterations_to_run = checkpoint_iter - (0 if len(strategies_over_time) == 0 else checkpoint_iterations[len(strategies_over_time)-1])
        
        # Run CFR for additional iterations up to this checkpoint
        cfr.iterations = iterations_to_run
        cfr.cfr_iterations_external()
        
        # Store a deep copy of the current strategies
        current_strategies = {}
        for info, node in cfr.nodes.items():
            current_strategies[info] = node.get_average_strategy().copy()
        
        strategies_over_time[checkpoint_iter] = current_strategies
    
    return strategies_over_time

def analyze_deck_size_impact(iterations=10000, deck_sizes=[2, 3, 4, 5], runs=3):
    """
    Analyze the impact of deck size on CFR performance.
    
    Args:
        iterations: Number of iterations for each run
        deck_sizes: List of deck sizes to evaluate
        runs: Number of runs for each deck size
    
    Returns:
        DataFrame with performance metrics for different deck sizes
    """
    results = []
    
    for deck_size in tqdm(deck_sizes, desc="Testing different deck sizes"):
        for run in range(runs):
            start_time = time.time()
            
            # Initialize and run CFR
            cfr = KuhnCFR(iterations, deck_size)
            cfr.cfr_iterations_external()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Calculate number of information sets
            num_infosets = len(cfr.nodes)
            
            # Group strategies by card
            card_strategies = defaultdict(list)
            for info, node in cfr.nodes.items():
                if len(info) > 0:  # Ensure info is not empty
                    card = info[0]
                    card_strategies[card].append(node.get_average_strategy())
            
            # Calculate average strategy variance per card
            card_variances = {}
            for card, strategies in card_strategies.items():
                if strategies:
                    stacked = np.vstack(strategies)
                    variance = np.mean(np.var(stacked, axis=0))
                    card_variances[card] = variance
            
            avg_variance = np.mean(list(card_variances.values())) if card_variances else 0
            
            results.append({
                'deck_size': deck_size,
                'run': run,
                'time': elapsed_time,
                'num_infosets': num_infosets,
                'strategy_variance': avg_variance
            })
    
    return pd.DataFrame(results)

def plot_convergence_results(results_df):
    """
    Create plots showing convergence metrics over iterations.
    """
    # Group by iterations and calculate means
    grouped = results_df.groupby('iterations').mean().reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot execution time vs iterations
    axes[0, 0].plot(grouped['iterations'], grouped['time'], 'o-', linewidth=2)
    axes[0, 0].set_title('Execution Time vs. Iterations', fontsize=14)
    axes[0, 0].set_xlabel('Iterations', fontsize=12)
    axes[0, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[0, 0].set_xscale('log')
    
    # Plot number of information sets vs iterations
    axes[0, 1].plot(grouped['iterations'], grouped['num_infosets'], 'o-', linewidth=2)
    axes[0, 1].set_title('Number of Information Sets vs. Iterations', fontsize=14)
    axes[0, 1].set_xlabel('Iterations', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_xscale('log')
    
    # Plot average entropy vs iterations
    axes[1, 0].plot(grouped['iterations'], grouped['avg_entropy'], 'o-', linewidth=2)
    axes[1, 0].set_title('Strategy Entropy vs. Iterations', fontsize=14)
    axes[1, 0].set_xlabel('Iterations', fontsize=12)
    axes[1, 0].set_ylabel('Entropy', fontsize=12)
    axes[1, 0].set_xscale('log')
    
    # Plot iterations/second vs iterations
    iters_per_sec = grouped['iterations'] / grouped['time']
    axes[1, 1].plot(grouped['iterations'], iters_per_sec, 'o-', linewidth=2)
    axes[1, 1].set_title('Iterations per Second vs. Iterations', fontsize=14)
    axes[1, 1].set_xlabel('Iterations', fontsize=12)
    axes[1, 1].set_ylabel('Iterations/Second', fontsize=12)
    axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('cfr_convergence_analysis.png', dpi=300)
    plt.close()

def plot_strategy_evolution(strategies_over_time, key_infosets=None):
    """
    Plot how strategies evolve for key information sets.
    
    Args:
        strategies_over_time: Dictionary from analyze_strategy_evolution
        key_infosets: List of specific information sets to plot (if None, will select some)
    """
    # If no key infosets provided, select some important ones
    if key_infosets is None:
        # Find most common infosets across all checkpoints
        all_infosets = set()
        for checkpoint, strategies in strategies_over_time.items():
            all_infosets.update(strategies.keys())
        
        # Filter for infosets with cards and different history lengths (simple heuristic)
        filtered_infosets = [info for info in all_infosets if len(info) >= 2]
        
        # Select a manageable number of diverse infosets
        if len(filtered_infosets) > 6:
            # Sort by card first, then history length
            sorted_infosets = sorted(filtered_infosets, key=lambda x: (x[0], len(x)))
            # Select with some spacing to get diversity
            key_infosets = sorted_infosets[::max(1, len(sorted_infosets)//6)][:6]
        else:
            key_infosets = filtered_infosets
    
    # Create subplots for each key infoset
    fig, axes = plt.subplots(len(key_infosets), 1, figsize=(12, 4*len(key_infosets)))
    if len(key_infosets) == 1:
        axes = [axes]
    
    checkpoints = sorted(strategies_over_time.keys())
    
    for i, infoset in enumerate(key_infosets):
        ax = axes[i]
        
        # Track strategy probabilities over checkpoints
        action_probs = {0: [], 1: []}
        
        for checkpoint in checkpoints:
            if infoset in strategies_over_time[checkpoint]:
                strategy = strategies_over_time[checkpoint][infoset]
                action_probs[0].append(strategy[0])
                action_probs[1].append(strategy[1])
            else:
                # If infoset not found at this checkpoint, use None as placeholder
                action_probs[0].append(None)
                action_probs[1].append(None)
        
        # Plot evolution
        ax.plot(checkpoints, action_probs[0], 'o-', label='Check/Fold', linewidth=2)
        ax.plot(checkpoints, action_probs[1], 's-', label='Bet/Call', linewidth=2)
        
        # Interpret the infoset string for labeling
        card = infoset[0]
        history = infoset[1:] if len(infoset) > 1 else ""
        history_str = " with history: " + history if history else " (initial)"
        
        ax.set_title(f'Strategy Evolution for Card {card}{history_str}', fontsize=14)
        ax.set_xlabel('Iterations', fontsize=12)
        ax.set_ylabel('Action Probability', fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('strategy_evolution.png', dpi=300)
    plt.close()

def plot_deck_size_impact(results_df):
    """
    Plot the impact of deck size on various performance metrics.
    """
    # Group by deck size and calculate means and std devs
    grouped = results_df.groupby('deck_size').agg({
        'time': ['mean', 'std'],
        'num_infosets': ['mean', 'std'],
        'strategy_variance': ['mean', 'std']
    }).reset_index()
    
    # Flatten the multi-level column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot execution time vs deck size
    axes[0].errorbar(
        grouped['deck_size'], 
        grouped['time_mean'], 
        yerr=grouped['time_std'],
        fmt='o-', 
        capsize=5,
        linewidth=2
    )
    axes[0].set_title('Execution Time vs. Deck Size', fontsize=14)
    axes[0].set_xlabel('Deck Size', fontsize=12)
    axes[0].set_ylabel('Time (seconds)', fontsize=12)
    
    # Plot number of info sets vs deck size
    axes[1].errorbar(
        grouped['deck_size'], 
        grouped['num_infosets_mean'], 
        yerr=grouped['num_infosets_std'],
        fmt='o-', 
        capsize=5,
        linewidth=2
    )
    axes[1].set_title('Number of Information Sets vs. Deck Size', fontsize=14)
    axes[1].set_xlabel('Deck Size', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    
    # Plot strategy variance vs deck size
    axes[2].errorbar(
        grouped['deck_size'], 
        grouped['strategy_variance_mean'], 
        yerr=grouped['strategy_variance_std'],
        fmt='o-', 
        capsize=5,
        linewidth=2
    )
    axes[2].set_title('Strategy Variance vs. Deck Size', fontsize=14)
    axes[2].set_xlabel('Deck Size', fontsize=12)
    axes[2].set_ylabel('Variance', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('deck_size_impact.png', dpi=300)
    plt.close()

def visualize_strategies(cfr, title='Final Strategies'):
    """
    Create a visualization of the final strategies for each information set.
    """
    # Group information sets by card
    card_infosets = defaultdict(list)
    for info, node in cfr.nodes.items():
        if len(info) > 0:  # Ensure info is not empty
            card = info[0]
            history = info[1:] if len(info) > 1 else ""
            strategy = node.get_average_strategy()
            card_infosets[card].append((history, strategy))
    
    # Sort cards
    cards = sorted(card_infosets.keys())
    
    # Create figure with subplots for each card
    fig, axes = plt.subplots(len(cards), 1, figsize=(12, 4*len(cards)))
    if len(cards) == 1:
        axes = [axes]
    
    for i, card in enumerate(cards):
        ax = axes[i]
        
        # Sort infosets by history length and content
        infosets = sorted(card_infosets[card], key=lambda x: (len(x[0]), x[0]))
        
        # Prepare data for grouped bar chart
        histories = [item[0] if item[0] else "Initial" for item in infosets]
        check_fold_probs = [item[1][0] for item in infosets]
        bet_call_probs = [item[1][1] for item in infosets]
        
        x = np.arange(len(histories))
        width = 0.35
        
        # Create grouped bar chart
        ax.bar(x - width/2, check_fold_probs, width, label='Check/Fold')
        ax.bar(x + width/2, bet_call_probs, width, label='Bet/Call')
        
        ax.set_title(f'Strategies for Card {card}', fontsize=14)
        ax.set_xlabel('Game History', fontsize=12)
        ax.set_ylabel('Action Probability', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(histories)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, axis='y')
        
        # Add probability values on top of bars
        for j, v in enumerate(check_fold_probs):
            ax.text(j - width/2, v + 0.02, f'{v:.2f}', ha='center')
        for j, v in enumerate(bet_call_probs):
            ax.text(j + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('cfr_strategies_visualization.png', dpi=300)
    plt.close()

def nash_equilibrium_analysis(iterations=100000, decksize=3):
    """
    Analyze how close the final strategy is to Nash equilibrium.
    For Kuhn poker, we have theoretical optimal strategies to compare against.
    """
    # Theoretical optimal strategy for Kuhn poker with 3 cards (simplified version)
    # These values are approximate and based on literature for standard Kuhn poker
    theoretical_optimal = {
        # Format: infoset -> [check/fold probability, bet/call probability]
        '0': [2/3, 1/3],  # Card 0, no history
        '1': [1, 0],      # Card 1, no history
        '2': [1/3, 2/3],  # Card 2, no history
        '01': [1, 0],     # Card 0, opponent bet
        '11': [2/3, 1/3], # Card 1, opponent bet
        '21': [0, 1],     # Card 2, opponent bet
        '00': [2/3, 1/3], # Card 0, both checked
        '10': [1, 0],     # Card 1, both checked
        '20': [0, 1]      # Card 2, both checked
    }
    
    # Run CFR
    cfr = KuhnCFR(iterations, decksize)
    cfr.cfr_iterations_external()
    
    # Calculate KL divergence between computed and theoretical strategies
    results = []
    for infoset, theoretical in theoretical_optimal.items():
        if infoset in cfr.nodes:
            computed = cfr.nodes[infoset].get_average_strategy()
            
            # KL divergence: sum(p * log(p/q))
            kl_div = 0
            for i in range(2):
                if theoretical[i] > 0 and computed[i] > 0:
                    kl_div += theoretical[i] * np.log(theoretical[i] / computed[i])
            
            # L1 distance: sum(|p - q|)
            l1_dist = sum(abs(theoretical[i] - computed[i]) for i in range(2))
            
            results.append({
                'infoset': infoset,
                'theoretical': theoretical,
                'computed': computed.tolist(),
                'kl_divergence': kl_div,
                'l1_distance': l1_dist
            })
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by infoset for consistent ordering
    results_df = results_df.sort_values('infoset')
    
    # Create grouped bar chart comparing theoretical and computed
    x = np.arange(len(results_df))
    width = 0.2
    
    # Theoretical probabilities
    ax.bar(x - 1.5*width, [t[0] for t in results_df['theoretical']], width, label='Theoretical Check/Fold')
    ax.bar(x - 0.5*width, [t[1] for t in results_df['theoretical']], width, label='Theoretical Bet/Call')
    
    # Computed probabilities
    ax.bar(x + 0.5*width, [c[0] for c in results_df['computed']], width, label='Computed Check/Fold')
    ax.bar(x + 1.5*width, [c[1] for c in results_df['computed']], width, label='Computed Bet/Call')
    
    ax.set_title('Comparison of Theoretical and Computed Nash Equilibrium Strategies', fontsize=14)
    ax.set_xlabel('Information Set', fontsize=12)
    ax.set_ylabel('Action Probability', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['infoset'])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('nash_equilibrium_comparison.png', dpi=300)
    plt.close()
    
    # Return dataframe for further analysis
    return results_df

if __name__ == "__main__":
    print("CFR Poker AI Testing and Evaluation")
    print("===================================")
    
    # 1. Test convergence over different iteration counts
    print("\n1. Testing convergence over different iteration counts...")
    iterations_list = [100, 500, 1000, 5000, 10000, 50000]
    convergence_results = evaluate_convergence(iterations_list, decksize=3, runs=3)
    plot_convergence_results(convergence_results)
    print("Convergence analysis complete. See cfr_convergence_analysis.png")
    
    # 2. Analyze strategy evolution
    print("\n2. Analyzing strategy evolution...")
    strategies_evolution = analyze_strategy_evolution(max_iterations=50000, checkpoints=10)
    plot_strategy_evolution(strategies_evolution)
    print("Strategy evolution analysis complete. See strategy_evolution.png")
    
    # 3. Analyze impact of deck size
    print("\n3. Analyzing impact of deck size...")
    deck_size_results = analyze_deck_size_impact(iterations=5000, deck_sizes=[2, 3, 4, 5], runs=3)
    plot_deck_size_impact(deck_size_results)
    print("Deck size impact analysis complete. See deck_size_impact.png")
    
    # 4. Visualize final strategies
    print("\n4. Visualizing final strategies...")
    final_cfr = KuhnCFR(50000, 3)
    final_cfr.cfr_iterations_external()
    visualize_strategies(final_cfr, title='Final Strategies after 50,000 Iterations')
    print("Strategy visualization complete. See cfr_strategies_visualization.png")
    
    # 5. Nash equilibrium analysis
    print("\n5. Performing Nash equilibrium analysis...")
    nash_results = nash_equilibrium_analysis(iterations=50000)
    
    # Calculate overall deviation metrics
    avg_kl = nash_results['kl_divergence'].mean()
    avg_l1 = nash_results['l1_distance'].mean()
    print(f"Average KL divergence from Nash equilibrium: {avg_kl:.4f}")
    print(f"Average L1 distance from Nash equilibrium: {avg_l1:.4f}")
    print("Nash equilibrium analysis complete. See nash_equilibrium_comparison.png")
    
    print("\nAll evaluations complete!")