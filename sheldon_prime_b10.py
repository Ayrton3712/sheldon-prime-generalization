def get_prime(n):
    i = 0
    for j in range(n - 1):
        if (n == 0):
            return 1
        
        divisor = 2
        while (divisor * divisor <= n):
            if (n % divisor == 0):
                continue
            divisor += 1

def reverse_digits(r):
    pass

def multiply_digit(r):
    pass

def is_sheldon(r):
    pass

#get_prime(3)