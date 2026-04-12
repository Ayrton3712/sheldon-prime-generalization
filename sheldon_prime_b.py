import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def sieve(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes.
    
    Returns a list of primes: primes[0]=2, primes[1]=3, etc.
    """
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    return primes


# Global cache: primes list and prime-to-index mapping
_primes = []
_prime_to_index = {}

def init_primes(limit=1000000):
    """Initialize the prime cache with all primes up to limit.
    
    Args:
        limit: Upper bound for sieve (generates all primes <= limit)
    """
    global _primes, _prime_to_index
    _primes = sieve(limit)
    _prime_to_index = {p: i + 1 for i, p in enumerate(_primes)}  # 1-indexed


def init_primes_by_count(count):
    """Initialize the prime cache with the first 'count' primes.
    
    Uses the prime number theorem to estimate the necessary limit:
    the nth prime is approximately n * ln(n) for large n.
    
    Args:
        count: Number of primes to generate
    
    Returns:
        The actual number of primes generated
    """
    import math
    
    if count < 10:
        limit = 30  # Handle small cases
    elif count < 100:
        limit = 550
    else:
        # Prime number theorem approximation: nth prime ~ n * ln(n)
        # Add 15% buffer to be safe
        limit = int(count * (math.log(count) + math.log(math.log(count))) * 1.15)
    
    # Keep increasing limit until we have enough primes
    while True:
        init_primes(limit)
        if len(_primes) >= count:
            return len(_primes)
        limit = int(limit * 1.5)  # Increase by 50% and try again


def get_prime(n):
    """Return the nth prime number (1-indexed).
    
    Uses precomputed sieve; requires init_primes() to be called first.
    Raises IndexError if n exceeds the precomputed range.
    """
    if n <= 0:
        return None
    if not _primes:
        raise IndexError("Prime cache not initialized. Call init_primes(limit) first.")
    if n <= len(_primes):
        return _primes[n - 1]
    raise IndexError(f"Prime index {n} exceeds precomputed range (limit covers {len(_primes)} primes). Call init_primes() with a higher limit.")


def get_digits_in_base(r, b):
    """Get the list of digits of r in base b."""
    if r == 0:
        return [0]
    digits = []
    while r > 0:
        digits.append(r % b)
        r //= b
    return digits[::-1]


def multiply_digits_in_base(r, b):
    """Return the product of the digits of r in base b.
    
    Short-circuits to 0 if any digit is 0 (matching never equal any valid n >= 1).
    """
    digits = get_digits_in_base(r, b)
    product = 1
    for digit in digits:
        if digit == 0:
            return 0
        product *= digit
    return product


def reverse_digits_in_base(r, b):
    """Return the number whose base-b representation is the reverse of r's base-b representation."""
    digits = get_digits_in_base(r, b)
    digits.reverse()
    result = 0
    for digit in digits:
        result = result * b + digit
    return result


def check_product_property(r, n, b):
    """Check if the product of digits of r in base b equals n."""
    return multiply_digits_in_base(r, b) == n


def check_mirror_property(r, n, b):
    """Check if the reverse of r in base b is the prime at position reverse_digits(n) in base b.
    
    Returns False if the reversed index exceeds the precomputed prime range.
    """
    reversed_n = reverse_digits_in_base(n, b)
    reversed_r = reverse_digits_in_base(r, b)
    try:
        return reversed_r == get_prime(reversed_n)
    except IndexError:
        # Reversed index exceeds precomputed range, so mirror property cannot be satisfied
        return False


def is_sheldon(r, b):
    """Check if r is a Sheldon prime in base b.
    
    A Sheldon prime must satisfy:
    - It is prime at position n
    - The product of its digits in base b equals n
    - Its reverse in base b is the prime at position reverse_digits(n) in base b
    """
    if r <= 0 or not _prime_to_index:
        return False
    
    # O(1) lookup: find n such that get_prime(n) == r
    if r not in _prime_to_index:
        return False  # r is not in our precomputed primes
    
    n = _prime_to_index[r]
    
    # Check both properties in base b
    return check_product_property(r, n, b) and check_mirror_property(r, n, b)


def is_sheldon_by_index(n, b):
    """Check if the nth prime is a Sheldon prime in base b.
    
    More efficient than is_sheldon(r, b) when iterating by prime index.
    """
    r = get_prime(n)
    if r is None:
        return False
    return check_product_property(r, n, b) and check_mirror_property(r, n, b)


def find_sheldon_primes(b, limit_index):
    """Return all Sheldon primes in base b among the first limit_index primes.
    
    Args:
        b: The base to check in (2-36)
        limit_index: Maximum prime index to check (1-indexed)
    
    Returns:
        List of primes that are Sheldon in base b, sorted.
    """
    return [_primes[n - 1] for n in range(1, limit_index + 1) if is_sheldon_by_index(n, b)]


def analyze_properties(b, limit_index):
    """Analyze property satisfaction for the first limit_index primes in base b.
    
    Args:
        b: The base to check in (2-36)
        limit_index: Maximum prime index to check (1-indexed)
    
    Returns:
        Dict with keys:
        - 'both': primes with both mirror and product properties (Sheldon)
        - 'product_only': primes with product property only
        - 'mirror_only': primes with mirror property only
        - 'neither': primes with neither property
        - 'data': pandas DataFrame with detailed analysis
    """
    
    data = []
    for n in range(1, limit_index + 1):
        r = get_prime(n)
        if r is None:
            break
        
        has_product = check_product_property(r, n, b)
        has_mirror = check_mirror_property(r, n, b)
        
        category = 'neither'
        if has_product and has_mirror:
            category = 'both'
        elif has_product:
            category = 'product_only'
        elif has_mirror:
            category = 'mirror_only'
        
        data.append({
            'index': n,
            'prime': r,
            'product_property': has_product,
            'mirror_property': has_mirror,
            'category': category
        })
    
    df = pd.DataFrame(data)
    
    return {
        'both': df[df['category'] == 'both']['prime'].tolist(),
        'product_only': df[df['category'] == 'product_only']['prime'].tolist(),
        'mirror_only': df[df['category'] == 'mirror_only']['prime'].tolist(),
        'neither': df[df['category'] == 'neither']['prime'].tolist(),
        'data': df
    }


def visualize_properties(b, limit_index, figsize=(14, 12), save_filename=None):
    """Create comprehensive visualizations of property distributions in base b.
    
    Args:
        b: The base to check in (2-36)
        limit_index: Maximum prime index to check (1-indexed)
        figsize: Figure size as (width, height) tuple
        save_filename: If provided, save the figure to this file path; otherwise return the figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Analyze properties
    analysis = analyze_properties(b, limit_index)
    df = analysis['data']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Property Distribution in Base {b} (First {limit_index} Primes)', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: prime value vs index, colored by category
    ax1 = axes[0, 0]
    colors = {'both': '#2ecc71', 'product_only': '#3498db', 'mirror_only': '#e74c3c', 'neither': '#95a5a6'}
    for category in ['product_only', 'mirror_only', 'both', 'neither']:
        subset = df[df['category'] == category]
        ax1.scatter(subset['index'], subset['prime'], label=category, alpha=0.7, s=50, color=colors[category])
    ax1.set_xlabel('Prime Index (n)', fontsize=11)
    ax1.set_ylabel('Prime Value', fontsize=11)
    ax1.set_title('Prime Value vs Index by Category', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Bar chart: count by category
    ax2 = axes[0, 1]
    category_counts = df['category'].value_counts()
    category_order = ['both', 'product_only', 'mirror_only', 'neither']
    category_counts = category_counts.reindex(category_order, fill_value=0)
    bars = ax2.bar(range(len(category_counts)), category_counts.values, color=[colors[cat] for cat in category_order])
    ax2.set_xticks(range(len(category_counts)))
    ax2.set_xticklabels(category_order, rotation=45, ha='right')
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distribution by Category', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, category_counts.values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(int(count)), 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Histogram: distribution of prime values by category
    ax3 = axes[1, 0]
    # Calculate shared bins based on all data to ensure consistent, visible widths
    all_primes_with_properties = []
    for category in ['both', 'product_only', 'mirror_only']:
        subset = df[df['category'] == category]['prime'].values
        if len(subset) > 0:
            all_primes_with_properties.extend(subset)
    
    # Use adaptive bin sizing: 20 bins or fewer bins if data is sparse
    if len(all_primes_with_properties) > 0:
        data_range = max(all_primes_with_properties) - min(all_primes_with_properties)
        num_bins = max(10, min(30, len(all_primes_with_properties) // 2))
        bin_edges = np.linspace(min(all_primes_with_properties), max(all_primes_with_properties), num_bins + 1)
    else:
        bin_edges = 30
    
    # Plot histograms with shared bins
    for category, color in [('both', colors['both']), ('product_only', colors['product_only']), ('mirror_only', colors['mirror_only'])]:
        subset = df[df['category'] == category]['prime'].values
        if len(subset) > 0:
            ax3.hist(subset, bins=bin_edges, alpha=0.6, label=category, color=color, edgecolor='black', linewidth=0.8)
    
    ax3.set_xlabel('Prime Value', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Prime Value Distribution (Properties Found)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Property co-occurrence heatmap
    ax4 = axes[1, 1]
    
    # Create a simpler 2x2 co-occurrence matrix
    cooccurrence = pd.DataFrame([
        [
            df[(df['product_property']) & (df['mirror_property'])].shape[0],  # Both
            df[(df['product_property']) & (~df['mirror_property'])].shape[0]   # Product only
        ],
        [
            df[(~df['product_property']) & (df['mirror_property'])].shape[0],  # Mirror only
            df[(~df['product_property']) & (~df['mirror_property'])].shape[0]  # Neither
        ]
    ], index=['Product=Yes', 'Product=No'], columns=['Mirror=Yes', 'Mirror=No'])
    
    sns.heatmap(cooccurrence, annot=True, fmt='d', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Count'})
    ax4.set_title('Property Co-occurrence Matrix', fontsize=12, fontweight='bold')
    
    # Increase space between top and bottom rows
    plt.subplots_adjust(hspace=0.55, wspace=0.3)
    
    # Save or return figure
    if save_filename:
        fig.savefig(save_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return analysis
    else:
        return fig, analysis


def print_property_results(b, limit_index):
    """Print detailed results of property analysis for base b.
    
    Args:
        b: The base to check in (2-36)
        limit_index: Maximum prime index to check (1-indexed)
    """
    analysis = analyze_properties(b, limit_index)
    
    print(f"\n{'='*70}")
    print(f"Property Analysis Results for Base {b} (First {limit_index} Primes)")
    print(f"{'='*70}\n")
    
    print(f"Sheldon Primes (Both Properties) [{len(analysis['both'])} total]:")
    print(f"  {analysis['both']}\n")
    
    print(f"Product Property Only [{len(analysis['product_only'])} total]:")
    print(f"  {analysis['product_only']}\n")
    
    print(f"Mirror Property Only [{len(analysis['mirror_only'])} total]:")
    print(f"  {analysis['mirror_only']}\n")
    
    print(f"Neither Property [{len(analysis['neither'])} total]")
    if len(analysis['neither']) <= 20:
        print(f"  {analysis['neither']}")
    else:
        # print(f"  {analysis['neither'][:20]} ... (showing first 20 of {len(analysis['neither'])})")
        pass # No interest in neither!!
    
    print(f"\n{'='*70}\n")


def estimate_required_prime_count(prime_count, bases):
    """Estimate the number of primes needed to check mirror property.
    
    When checking the mirror property, reversed_n = reverse_digits_in_base(n, b)
    can be much larger than n, especially in small bases. This function estimates
    the maximum reversed index that could occur and returns the required prime count.
    
    Args:
        prime_count: Number of primes we want to analyze
        bases: List of bases to check
    
    Returns:
        The number of primes to generate to ensure all mirror property checks are valid
    """
    max_reversed_n = 0
    
    for b in bases:
        # Sample indices across the range to estimate maximum reversed index
        # We'll check indices at different scales
        test_indices = set()
        
        # Add some small indices
        test_indices.update(range(1, min(1000, prime_count)))
        
        # Add regularly spaced indices across the range
        if prime_count > 1000:
            for multiplier in [100, 1000, 10000, 100000, 1000000]:
                for scale in [1, 2, 5, 9]:
                    idx = (scale * multiplier)
                    if idx <= prime_count:
                        test_indices.add(idx)
            # Add the max index
            test_indices.add(prime_count)
        
        # Compute reversed index for each test index
        for n in test_indices:
            reversed_n = reverse_digits_in_base(n, b)
            max_reversed_n = max(max_reversed_n, reversed_n)
    
    # Add a 10% safety margin
    required = int(max_reversed_n * 1.1)
    return max(prime_count, required)


def analyze_all_bases(prime_count, bases, verbose=True, save_figures=True):
    """Analyze and print property distributions across multiple bases.
    
    Args:
        prime_count: Number of primes to analyze (e.g., 10_000_000)
        bases: List of bases to check (e.g., [3, 4, 5, ..., 16])
        verbose: If True, print results for each base; if False, only show summary
        save_figures: If True, save visualization PNG files for each base
    
    Returns:
        Dict mapping base -> analysis results
    """
    import os
    
    print(f"\n{'='*80}")
    print(f"Analyzing first {prime_count:,} primes across bases {bases}")
    print(f"Estimating required prime cache size to check mirror property...")
    
    # Estimate how many primes we need to generate to handle mirror property checks
    required_count = estimate_required_prime_count(prime_count, bases)
    
    if required_count > prime_count:
        print(f"Mirror property checks require {required_count:,} primes (vs {prime_count:,} to analyze)")
        print(f"Generating {required_count:,} primes...")
    else:
        print(f"Initializing prime cache for {prime_count:,} primes...")
    
    print(f"{'='*80}\n")
    
    actual_count = init_primes_by_count(required_count)
    print(f"Generated {actual_count:,} primes\n")
    
    results = {}
    for b in bases:
        print(f"Analyzing base {b}...")
        # Analyze only the first prime_count primes (not the extra ones needed for mirror property)
        analysis = analyze_properties(b, prime_count)
        results[b] = analysis
        
        if verbose:
            print_property_results(b, prime_count)
        else:
            # Compact summary
            print(f"  Sheldon: {len(analysis['both'])}, Product only: {len(analysis['product_only'])}, Mirror only: {len(analysis['mirror_only'])}")
        
        # Save figure for this base
        if save_figures:
            filename = f"base_{b}_properties.png"
            print(f"  Saving figure to {filename}...")
            visualize_properties(b, prime_count, save_filename=filename)
            print(f"  ✓ Saved {filename}")
    
    print(f"\n{'='*80}")
    print(f"Analysis complete. {len(bases)} PNG files saved to current directory.")
    print(f"{'='*80}\n")
    
    return results

# Analyze first 10 million primes in bases 3-16 (except 10)
if __name__ == "__main__":
    prime_count = 10_000_000
    bases = [b for b in range(3, 17) if b != 10]  # 3-16, excluding 10
    
    # This will initialize once for all bases
    results = analyze_all_bases(prime_count, bases, verbose=True)   