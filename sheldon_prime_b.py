def get_prime(n):
    """Return the nth prime number (1-indexed)."""
    if n <= 0:
        return None
    if n == 1:
        return 2
    
    count = 1  # We already have 2 as the first prime
    candidate = 3
    
    while True:
        # Check if candidate is prime
        is_prime = True
        divisor = 2
        while divisor * divisor <= candidate:
            if candidate % divisor == 0:
                is_prime = False
                break
            divisor += 1
        
        if is_prime:
            count += 1
            if count == n:
                return candidate
        
        candidate += 2  # Only check odd numbers


def convert_to_base(b, n):
    """Convert integer n to base b and return as an integer."""
    if n == 0:
        return 0
    
    digits = []
    while n > 0:
        digits.append(n % b)
        n //= b
    
    # Reverse to get the correct order (most significant digit first)
    digits.reverse()
    
    # Convert the digit list to an integer representation
    result = 0
    for digit in digits:
        result = result * 10 + digit
    
    return result


def reverse_digits(r):
    """Return the number with its digits reversed."""
    return int(str(r)[::-1])


def multiply_digits(r):
    """Return the product of the digits of r."""
    product = 1
    for digit in str(r):
        product *= int(digit)
    return product


def check_product_property(r, n):
    """Check if the product of digits of r equals n."""
    return multiply_digits(r) == n


def check_mirror_property(r, n):
    """Check if the reverse of r is the prime at position reverse_digits(n)."""
    reversed_n = reverse_digits(n)
    reversed_r = reverse_digits(r)
    return reversed_r == get_prime(reversed_n)


def is_sheldon(r):
    """Check if r is a Sheldon prime.
    
    A Sheldon prime must satisfy:
    - It is prime at position n
    - The product of its digits equals n
    - Its reverse is the prime at position reverse_digits(n)
    """
    if r <= 0:
        return False
    
    # Find n such that get_prime(n) == r
    n = 1
    while True:
        prime = get_prime(n)
        if prime == r:
            break
        if prime > r:
            # r is not prime
            return False
        n += 1
    
    # Check both properties
    return check_product_property(r, n) and check_mirror_property(r, n)

print(is_sheldon(73))