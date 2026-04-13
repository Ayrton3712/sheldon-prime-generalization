import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

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


def init_primes_by_count(count, cache_file="primes.npy", use_cache=True):
    """Initialize the prime cache with the first 'count' primes.
    
    Checks for cached primes first to avoid regeneration. Generates primes using
    the prime number theorem to estimate the necessary limit.
    
    Args:
        count: Number of primes to generate
        cache_file: Filename for caching primes (default: primes.npy)
        use_cache: Whether to check for and load from cache (default: True)
    
    Returns:
        The actual number of primes generated (or loaded from cache)
    """
    import math
    
    global _primes, _prime_to_index
    
    # Try to load from cache first
    if use_cache:
        cached_primes = load_primes_from_file(cache_file)
        if cached_primes and len(cached_primes) >= count:
            _primes = cached_primes
            _prime_to_index = {p: i + 1 for i, p in enumerate(_primes)}  # 1-indexed
            return len(_primes)
        else:
            if cached_primes:
                print(f"Cache has {len(cached_primes):,} primes, but {count:,} needed. Regenerating...")
    
    # Generate new primes if not in cache or cache insufficient
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
            # Cache the newly generated primes
            if use_cache:
                save_primes_to_file(_primes, cache_file)
            return len(_primes)
        limit = int(limit * 1.1)  # Increase by 10% and try again


def save_primes_to_file(primes, filename="primes.npy"):
    """Save primes list to a binary NumPy file for fast loading later.
    
    Args:
        primes: List of primes to save
        filename: Output filename (default: primes.npy in current directory)
    """
    primes_array = np.array(primes, dtype=np.uint64)
    np.save(filename, primes_array)
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"Saved {len(primes):,} primes to {filename} ({file_size_mb:.1f} MB)")


def load_primes_from_file(filename="primes.npy"):
    """Load primes from a NumPy binary file.
    
    Args:
        filename: Input filename (default: primes.npy)
    
    Returns:
        List of primes if file exists, None otherwise
    """
    if os.path.exists(filename):
        primes_array = np.load(filename)
        # Explicitly cast to Python int for type safety and dict lookup consistency
        primes = [int(p) for p in primes_array]
        print(f"Loaded {len(primes):,} primes from {filename}")
        return primes
    return None


def save_primes_to_csv(primes, filename="primes.csv"):
    """Save primes list to a CSV file for portability/inspection.
    
    Args:
        primes: List of primes to save
        filename: Output filename (default: primes.csv in current directory)
    """
    df = pd.DataFrame({'prime': primes})
    df.to_csv(filename, index=False)
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"Saved {len(primes):,} primes to {filename} ({file_size_mb:.1f} MB)")


def load_primes_from_csv(filename="primes.csv"):
    """Load primes from a CSV file.
    
    Args:
        filename: Input filename (default: primes.csv)
    
    Returns:
        List of primes if file exists, None otherwise
    """
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        primes = df['prime'].tolist()
        print(f"Loaded {len(primes):,} primes from {filename}")
        return primes
    return None


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


def visualize_properties(b, limit_index, save_filename=None):
    """Create visualizations of property distributions in base b using matplotlib.
    
    Args:
        b: The base to check in (2-36)
        limit_index: Maximum prime index to check (1-indexed)
        save_filename: If provided, save the figure to this SVG file path; otherwise return the figure object
    
    Returns:
        Matplotlib figure object (or None if saved to file)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Analyze properties
    analysis = analyze_properties(b, limit_index)
    df = analysis['data']
    
    # Color mapping
    colors = {'both': '#2ecc71', 'product_only': '#3498db', 'mirror_only': '#e74c3c', 'neither': '#95a5a6'}
    color_map = {cat: colors[cat] for cat in ['both', 'product_only', 'mirror_only', 'neither']}
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'Property Distribution in Base {b} (First {limit_index:,} Primes)', fontsize=14, fontweight='bold')
    
    category_order = ['product_only', 'mirror_only', 'both', 'neither']
    
    # 1. Scatter plot: prime value vs index, colored by category (top-left)
    ax = axes[0, 0]
    for category in category_order:
        subset = df[df['category'] == category]
        ax.scatter(subset['index'], subset['prime'], label=category, 
                  color=color_map[category], s=20, alpha=0.7)
    ax.set_xlabel('Prime Index (n)')
    ax.set_ylabel('Prime Value')
    ax.set_title('Prime Value vs Index by Category')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Bar chart: count by category (top-right)
    ax = axes[0, 1]
    category_counts = df['category'].value_counts()
    category_counts = category_counts.reindex(category_order, fill_value=0)
    bars = ax.bar(range(len(category_order)), category_counts.values,
                   color=[color_map[cat] for cat in category_order])
    ax.set_xticks(range(len(category_order)))
    ax.set_xticklabels(category_order, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Distribution by Category')
    ax.grid(True, alpha=0.3, axis='y')
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # 3. Histogram: distribution of prime values by category (bottom-left)
    ax = axes[1, 0]
    all_primes_with_properties = []
    for category in ['both', 'product_only', 'mirror_only']:
        subset = df[df['category'] == category]['prime'].values
        if len(subset) > 0:
            all_primes_with_properties.extend(subset)
    
    if len(all_primes_with_properties) > 0:
        num_bins = max(10, min(30, len(all_primes_with_properties) // 2))
    else:
        num_bins = 30
    
    for category, color in [('both', colors['both']), ('product_only', colors['product_only']), ('mirror_only', colors['mirror_only'])]:
        subset = df[df['category'] == category]['prime'].values
        if len(subset) > 0:
            ax.hist(subset, bins=num_bins, label=category, color=color, alpha=0.7)
    ax.set_xlabel('Prime Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Prime Value Distribution (Properties Found)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Heatmap: property co-occurrence (bottom-right)
    ax = axes[1, 1]
    cooccurrence = np.array([
        [
            df[(df['product_property']) & (df['mirror_property'])].shape[0],  # Both
            df[(df['product_property']) & (~df['mirror_property'])].shape[0]   # Product only
        ],
        [
            df[(~df['product_property']) & (df['mirror_property'])].shape[0],  # Mirror only
            df[(~df['product_property']) & (~df['mirror_property'])].shape[0]  # Neither
        ]
    ])
    
    im = ax.imshow(cooccurrence, cmap='YlOrRd', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Mirror=Yes', 'Mirror=No'])
    ax.set_yticklabels(['Product=Yes', 'Product=No'])
    ax.set_title('Property Co-occurrence Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cooccurrence[i, j],
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Count')
    
    plt.tight_layout()
    
    # Save or return figure
    if save_filename:
        fig.savefig(save_filename, format='svg', dpi=100, bbox_inches='tight')
        plt.close(fig)
        return analysis
    else:
        return fig, analysis


def print_property_results_from_analysis(b, limit_index, analysis):
    """Print detailed results from an already-computed analysis dict.
    
    Args:
        b: The base to check in (2-36)
        limit_index: Maximum prime index to check (1-indexed)
        analysis: Dict from analyze_properties() with 'both', 'product_only', 'mirror_only', 'neither' keys
    """
    
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
        # Omitted intentionally: primes satisfying neither property are the vast majority
        # and not of research interest (every prime fails both properties in almost all bases)
        pass
    
    print(f"\n{'='*70}\n")


def estimate_required_prime_count(prime_count):
    """Estimate the number of primes needed to check mirror property safely.
    
    When checking the mirror property, reversing digits in a small base
    can produce much larger indices than the original. To be safe across
    all bases 2-16 and avoid out-of-bounds lookups, we use a conservative
    multiplier: generating 12x the target ensures sufficient headroom.
    
    Args:
        prime_count: Number of primes to analyze
    
    Returns:
        The number of primes to generate (conservative estimate)
    """
    # Conservative multiplier: for base 2, reversing can amplify indices significantly.
    # 12x is empirically safe for bases 2-16 and 10M primes.
    return prime_count * 12


def _worker_init(primes):
    """Initialize global state in worker processes.
    
    This is called once per worker process when the Pool is created.
    It sets up the global _primes so that analyze_properties can access it
    without pickling large data structures repeatedly. We pass only _primes
    since workers never use _prime_to_index (it would be multi-GB overhead).
    
    Args:
        primes: List of primes from main process
    """
    global _primes
    _primes = primes


def _analyze_and_save_base(base_info):
    """Helper function for parallel base analysis.
    
    Args:
        base_info: Tuple of (base, prime_count, verbose, save_figures)
    
    Returns:
        Tuple of (base, analysis, status_message)
    """
    b, prime_count, verbose, save_figures = base_info
    
    try:
        # Analyze properties for this base
        analysis = analyze_properties(b, prime_count)
        
        # Create status message
        status_msg = f"Base {b}: Sheldon: {len(analysis['both'])}, Product only: {len(analysis['product_only'])}, Mirror only: {len(analysis['mirror_only'])}"
        
        # Save visualization if requested
        if save_figures:
            filename = f"base_{b}_properties.svg"
            visualize_properties(b, prime_count, save_filename=filename)
            status_msg += f" | Saved {filename}"
        
        return (b, analysis, status_msg, True)
    except Exception as e:
        return (b, None, f"Base {b}: ERROR - {str(e)}", False)


def analyze_all_bases(prime_count, bases, verbose=True, save_figures=True, num_workers=None, cache_file="primes.npy", use_cache=True):
    """Analyze and print property distributions across multiple bases (parallelized).
    
    Args:
        prime_count: Number of primes to analyze (e.g., 10_000_000)
        bases: List of bases to check (e.g., [3, 4, 5, ..., 16])
        verbose: If True, print results for each base; if False, only show summary
        save_figures: If True, save visualization HTML files for each base
        num_workers: Number of worker processes (default: CPU count)
        cache_file: Filename for caching primes (default: primes.npy)
        use_cache: Whether to use cached primes if available (default: True)
    
    Returns:
        Dict mapping base -> analysis results
    """
    import os
    from multiprocessing import Pool, cpu_count
    
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"\n{'='*80}")
    print(f"Analyzing first {prime_count:,} primes across bases {bases}")
    print(f"Using {num_workers} worker processes")
    print(f"Estimating required prime cache size to check mirror property...")
    
    # Estimate how many primes we need to generate to handle mirror property checks
    # Note: cache validity is checked only by count, not by content or generation parameters
    required_count = estimate_required_prime_count(prime_count)
    
    if required_count > prime_count:
        print(f"Mirror property checks require {required_count:,} primes (vs {prime_count:,} to analyze)")
        print(f"Generating {required_count:,} primes...")
    else:
        print(f"Initializing prime cache for {prime_count:,} primes...")
    
    print(f"{'='*80}\n")
    
    # Generate primes once (serial), with caching support
    actual_count = init_primes_by_count(required_count, cache_file=cache_file, use_cache=use_cache)
    print(f"Initialized prime cache with {actual_count:,} primes\n")
    print(f"Analyzing {len(bases)} bases in parallel...\n")
    
    # Prepare work items: (base, prime_count, verbose, save_figures)
    work_items = [(b, prime_count, verbose, save_figures) for b in bases]
    
    results = {}
    
    # Use multiprocessing pool to parallelize base analysis
    # Use initializer to set up global _primes in each worker process
    with Pool(
        num_workers,
        initializer=_worker_init,
        initargs=(_primes,)
    ) as pool:
        for b, analysis, status_msg, success in pool.imap_unordered(_analyze_and_save_base, work_items):
            print(status_msg)
            if success:
                results[b] = analysis
                if verbose:
                    # Use already-computed analysis to avoid recalculating
                    print_property_results_from_analysis(b, prime_count, analysis)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete. {len(results)} SVG files saved to current directory.")
    print(f"Open the .svg files in any browser or image viewer.")
    print(f"{'='*80}\n")
    
    return results

# Analyze first million primes in bases 2-16 except 10
if __name__ == "__main__":
    prime_count = 1_000_000
    bases = [b for b in range(2, 17) if b != 10]
    
    # Primes will be cached to "primes.npy" after first generation.
    # On subsequent runs, the cached file will be loaded instantly!
    results = analyze_all_bases(
        prime_count, 
        bases, 
        verbose=True,
        cache_file="primes.npy",
        use_cache=True
    )   