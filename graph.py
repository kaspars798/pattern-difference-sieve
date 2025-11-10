import matplotlib.pyplot as plt
import numpy as np

def pattern_difference_graph(limit=50):
    primes = [2, 3]
    sum_history = np.zeros(limit + 1, dtype=int)
    
    def apply_pattern(p):
        if p*p > limit:
            return
        pattern = np.full(p, p)
        pattern[0] = p - 1
        pos = p * p
        idx = 0
        while pos <= limit:
            sum_history[pos] += p - pattern[idx % p]
            idx += 1
            pos += 1
    
    # Build graph
    i = 0
    while primes[i]*primes[i] <= limit:
        p = primes[i]
        apply_pattern(p)
        
        # Find new primes in the safe zone
        start = p*p
        end = min(limit, (primes[i+1] if i+1 < len(primes) else limit+1)**2 - 1)
        for k in range(start, end):
            if sum_history[k] == sum_history[k-1]:
                candidate = k + 1
                if candidate not in primes:
                    primes.append(candidate)
        i += 1
    
    # Final pass
    for k in range(primes[-1]*primes[-1], limit):
        if sum_history[k] == sum_history[k-1]:
            candidate = k + 1
            if candidate not in primes:
                primes.append(candidate)
    
    # Plot
    x = np.arange(1, limit + 1)
    y = sum_history[1:]
    
    plt.figure(figsize=(14, 8))
    plt.step(x, y, where='post', linewidth=3, color='navy')
    
    # Color plateaus
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    current_y = 0
    start = 1
    for i in range(1, len(y)):
        if y[i] != current_y:
            plt.hlines(current_y, start, i, colors[current_y % len(colors)], linewidth=6)
            start = i
            current_y = y[i]
    plt.hlines(current_y, start, limit, colors[current_y % len(colors)], linewidth=6)
    
    # Mark primes
    for p in primes:
        if p <= limit:
            plt.scatter(p, sum_history[p], color='red', s=100, zorder=10, edgecolors='white', linewidth=2)
            plt.text(p, sum_history[p] + 0.3, str(p), ha='center', fontsize=10, fontweight='bold')
    
    plt.title("Pattern-Difference Sieve: Cumulative Sum Graph (n ≤ 50)", fontsize=18, pad=20)
    plt.xlabel("Number n", fontsize=14)
    plt.ylabel("Cumulative sum (number of marking primes ≤ √n)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, limit)
    plt.ylim(-0.5, max(y) + 1)
    plt.tight_layout()
    plt.savefig("pattern_difference_sieve_colored.pdf", dpi=300, bbox_inches='tight')
    plt.show()

pattern_difference_graph(50)
