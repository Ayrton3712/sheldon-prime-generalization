# Sheldon Primes in Arbitrary Bases

A computational exploration of Sheldon primes generalized to bases 2–16, extending the open problems posed by Pomerance & Spicer (2019).

## Background

A **Sheldon prime** is a prime $p_n$ (the $n$-th prime) satisfying two properties, originally defined in base 10:

- **Product property**: the product of the base-$b$ digits of $p_n$ equals $n$
- **Mirror property**: reversing the base-$b$ digits of $p_n$ yields the prime at the reversed index — i.e., $\text{rev}_b(p_n) = p_{\text{rev}_b(n)}$

In base 10, the only Sheldon prime is 73 (Pomerance & Spicer, 2019). This project asks: *do Sheldon primes exist in other bases?*

## Repository Structure

```
.
├── sheldon.py       # Main script
├── primes.npy       # Cached prime list (generated on first run)
└── base_*.svg       # Per-base visualization outputs
```

## How It Works

### 1. Prime Generation

Primes are generated once via the **Sieve of Eratosthenes** and cached to `primes.npy` for reuse across runs. To safely check the mirror property, the cache must cover indices larger than the analysis range — reversing digits in a small base can produce indices much larger than the original. A **12× multiplier** over the analysis range provides sufficient headroom for all bases 2–16.

```python
init_primes_by_count(required_count, cache_file="primes.npy")
```

On subsequent runs the sieve is skipped and primes are loaded from disk in seconds.

### 2. Property Checks

Both properties are computed purely from base-$b$ digit representations of the underlying integers. Primality itself is base-independent — only the digit-level operations change across bases.

```python
check_product_property(r, n, b)   # product of base-b digits of r == n
check_mirror_property(r, n, b)    # rev_b(r) == p[rev_b(n)]
```

`get_digits_in_base(r, b)` extracts digits via repeated modulo and integer division. The product check short-circuits to 0 on any zero digit, since no valid index $n \geq 1$ equals 0. The mirror check catches `IndexError` and returns `False` if the reversed index exceeds the cache range.

### 3. Analysis

`analyze_properties(b, limit_index)` iterates over the first `limit_index` primes and classifies each into one of four categories: `both` (Sheldon), `product_only`, `mirror_only`, or `neither`. It returns a summary dict and a full Pandas DataFrame for downstream inspection.

### 4. Parallelism

Each base is independent, so bases are analyzed concurrently using `multiprocessing.Pool`. The prime list is passed to workers once at pool creation via an **initializer function**, avoiding repeated pickling across tasks:

```python
Pool(num_workers, initializer=_worker_init, initargs=(_primes,))
```

Only `_primes` is shared — not the `prime→index` dict — since workers iterate by index and never need reverse lookups. Results stream back via `imap_unordered` so each base prints as it finishes.

### 5. Visualization

`visualize_properties(b, limit_index)` produces a 3-panel matplotlib figure per base (saved as SVG):

- **Bar chart**: count of primes in each category
- **Histogram**: distribution of prime values among property-satisfying primes
- **Heatmap**: co-occurrence matrix of the two properties

## Usage

```bash
pip install pandas numpy matplotlib

python sheldon.py
```

By default, analyzes the first **1,000,000 primes** across **bases 2–16** (excluding 10). Edit the `__main__` block to change the range:

```python
prime_count = 1_000_000
bases = [b for b in range(2, 17) if b != 10]
```

On first run, ~12M primes are generated and saved to `primes.npy` (~90 MB). Subsequent runs load from cache and proceed directly to analysis.

## Running on a Remote Server

For large runs, use `tmux` to keep the process alive across SSH disconnections:

```bash
tmux new -s sheldon
python sheldon.py 2>&1 | tee run_log.txt
# Detach: Ctrl+B, D
# Reattach later: tmux attach -t sheldon
```

## References

- Byrnes, J., Spicer, C., Turnquist, A. (2015). *The Sheldon Conjecture*. Math Horizons, 23(2), 12–15.
- Pomerance, C., Spicer, C. (2019). *Proof of the Sheldon Conjecture*. The American Mathematical Monthly, 126(8), 688–698.
