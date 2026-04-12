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
    """Initialize the prime cache with all primes up to limit."""
    global _primes, _prime_to_index
    _primes = sieve(limit)
    _prime_to_index = {p: i + 1 for i, p in enumerate(_primes)}  # 1-indexed


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
    """Check if the reverse of r in base b is the prime at position reverse_digits(n) in base b."""
    reversed_n = reverse_digits_in_base(n, b)
    reversed_r = reverse_digits_in_base(r, b)
    return reversed_r == get_prime(reversed_n)


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


def visualize_properties(b, limit_index, figsize=(14, 12)):
    """Create comprehensive visualizations of property distributions in base b.
    
    Args:
        b: The base to check in (2-36)
        limit_index: Maximum prime index to check (1-indexed)
        figsize: Figure size as (width, height) tuple
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
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
    import numpy as np
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


init_primes(1000000)
base_to_visualize = 10
primes_to_visualize = 100

# Print results
print_property_results(base_to_visualize, primes_to_visualize)

# Show visualization
fig, analysis = visualize_properties(base_to_visualize, primes_to_visualize)
plt.show()

print(f"Base {base_to_visualize} Analysis (first {primes_to_visualize} primes):")
print(f"  Sheldon primes (both): {len(analysis['both'])}")
print(f"  Product only: {len(analysis['product_only'])}")
print(f"  Mirror only: {len(analysis['mirror_only'])}")
print(f"  Neither: {len(analysis['neither'])}")


# Initial test
# init_primes(1000000)  # Initialize prime cache up to 1,000,000
# print(is_sheldon(73, 10))