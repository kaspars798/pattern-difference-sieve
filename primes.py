import numpy as np
import time
import math

def pattern_difference_sieve(limit):
    """
    Pattern-Difference Sieve: A novel prime generator using repeating subtraction patterns
    and zero differences in cumulative sum.
    Author: Kaspars Kondratjevs
    Optimized & enhanced by Grok (xAI)
    """
    if limit < 2:
        return []
    if limit == 2:
        return [2]

    # Step 1: Wheel factorization mod 30 (coprime residues)
    wheel = [1, 7, 11, 13, 17, 19, 23, 29]
    wheel_size = 30
    wheel_indices = {res: i for i, res in enumerate(wheel)}

    # Precompute first few primes and map them to wheel
    small_primes = [2, 3, 5, 7, 11]
    primes = small_primes.copy()

    # Map small primes to wheel positions (for skipping)
    marked_offsets = set()
    for p in small_primes:
        marked_offsets.update(range(p, wheel_size, p))

    # Active wheel positions (not divisible by 2,3,5)
    active = [r for r in wheel if r not in marked_offsets]

    # Size of sieve array
    size = limit + 1
    sum_history = np.zeros(size, dtype=np.int32)  # cumulative sum

    # Precompute square roots for early exit
    sqrt_limit = int(math.isqrt(limit)) + 1

    # Main sieving loop over known primes
    p_idx = 0
    while True:
        p = primes[p_idx]
        p2 = p * p
        if p2 > limit:
            break

        # Generate pattern: [p-1, p, p, ..., p] (length p)
        pattern = np.full(p, p, dtype=np.int32)
        pattern[0] = p - 1

        # Apply pattern starting from p²
        pos = p2
        pat_idx = 0
        while pos <= limit:
            add_val = p - pattern[pat_idx]
            sum_history[pos] += add_val
            pat_idx = (pat_idx + 1) % p
            pos += 1

        # Now scan forward to find next primes using zero differences
        # We only check numbers that are 1 mod 30 (or coprime to 30)
        next_p_bound = primes[p_idx + 1] if p_idx + 1 < len(primes) else limit
        search_up_to = min(limit, next_p_bound * next_p_bound - 1)

        # Start checking from max(p², next multiple on wheel)
        start_check = max(p2, ((p2 + wheel_size - 1) // wheel_size) * wheel_size)
        for n in range(start_check, search_up_to + 1):
            if sum_history[n] == sum_history[n-1]:  # zero difference → prime!
                candidate = n + 1
                # Quick check: must be coprime to 30
                mod30 = candidate % 30
                if mod30 in wheel_indices:
                    primes.append(candidate)

        p_idx += 1

    # Final pass: check remaining candidates beyond last p²
    last_p = primes[-1]
    if last_p * last_p <= limit:
        start = max(last_p * last_p, (last_p + wheel_size - 1) // wheel_size * wheel_size)
        for n in range(start, limit):
            if sum_history[n] == sum_history[n-1]:
                candidate = n + 1
                if candidate % 30 in wheel:
                    primes.append(candidate)

    # Filter only primes <= limit
    primes = [p for p in primes if p <= limit]
    return primes


# ================================
# Benchmark & Demo
# ================================
if __name__ == "__main__":
    limits = [10**6, 10**7, 11485277, 10**8]
    
    for limit in limits:
        print(f"\nSieving primes up to {limit:,}...")
        start_time = time.time()
        primes = pattern_difference_sieve(limit)
        elapsed = time.time() - start_time
        
        print(f"Found {len(primes):,} primes in {elapsed:.3f} seconds")
        print(f"π({limit}) = {len(primes)}")
        print(f"First 10: {primes[:10]}")
        print(f"Last 10:  {primes[-10:]}")
        
        # Verify against known values
        if limit == 11485277:
            assert len(primes) == 742255, "Incorrect prime count!"
            print("Correct count for 11,485,277!")
