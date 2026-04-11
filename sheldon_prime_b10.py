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

def reverse_digits(r):
    """Return the number with its digits reversed."""
    return int(str(r)[::-1])

def multiply_digit(r):
    """Return the product of the digits of r."""
    product = 1
    for digit in str(r):
        product *= int(digit)
    return product

def is_sheldon(r):
    pass

print(get_prime(21))