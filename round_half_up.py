# function that takes in a decimal number and rounds it to the nearest integer

def round_half_up(n, decimals):
    import math

    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier