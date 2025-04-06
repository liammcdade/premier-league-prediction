import os
import numpy as np
from utils import get_average_probability

def load_match_odds(match_name):
    """Load odds for a given match."""
    match_path = f"matches/{match_name}/odds.txt"
    if not os.path.exists(match_path):
        raise ValueError(f"Odds file for {match_name} not found.")
    
    with open(match_path, 'r') as f:
        lines = f.readlines()
        man_city_odds = []
        man_united_odds = []
        for line in lines:
            if line.startswith("man_city_odds"):
                man_city_odds = eval(line.split("=")[1].strip())
            if line.startswith("man_united_odds"):
                man_united_odds = eval(line.split("=")[1].strip())
        return man_city_odds, man_united_odds

def calculate_draw_odds(man_city_odds, man_united_odds):
    """Generate draw odds based on both teams' odds."""
    draw_odds = []
    for city_odds, united_odds in zip(man_city_odds, man_united_odds):
        city_prob = get_average_probability(city_odds)
        united_prob = get_average_probability(united_odds)
        draw_prob = 1 - (city_prob + united_prob)
        draw_odds.append(draw_prob)
    return draw_odds

def apply_randomness(final_probs, match_minute):
    randomness_factor = np.random.uniform(0.04, 0.095)
    if match_minute < 15:
        randomness_factor *= 0.8
    elif match_minute > 75:
        randomness_factor *= 1.2

    for team in final_probs:
        final_probs[team] += final_probs[team] * randomness_factor
    total_prob = sum(final_probs.values())
    for team in final_probs:
        final_probs[team] /= total_prob

    return final_probs

def simulate_match(final_probs, simulations=100_000):
    outcomes = {'Man City': 0, 'Draw': 0, 'Man United': 0}
    for _ in range(simulations):
        r = np.random.random()
        if r < final_probs['Man City']:
            outcomes['Man City'] += 1
        elif r < final_probs['Man City'] + final_probs['Draw']:
            outcomes['Draw'] += 1
        else:
            outcomes['Man United'] += 1

    total = simulations
    for k, v in outcomes.items():
        print(f"{k} Win Probability: {v / total * 100:.2f}%")

# Load odds for each match
match_name = "man-united-vs-man-city-2025"
man_city_odds, man_united_odds = load_match_odds(match_name)

# Example: Choose match 1 (you can modify it to handle dynamic input)
match = 0  # Assuming match 1 in the odds file

# Convert to probabilities for each team
man_city_prob = get_average_probability(man_city_odds[match])
man_united_prob = get_average_probability(man_united_odds[match])

# Calculate draw odds
draw_odds = calculate_draw_odds(man_city_odds, man_united_odds)

# Normalize the probabilities
total_prob = man_city_prob + man_united_prob + draw_odds[match]  # Taking first draw odds
man_city_prob /= total_prob
man_united_prob /= total_prob
draw_prob = draw_odds[match] / total_prob

odds_probs = {'Man City': man_city_prob, 'Man United': man_united_prob, 'Draw': draw_prob}

# Apply randomness based on match minute
match_minute = int(input("Enter the current minute of the match: "))
final_probs = apply_randomness(odds_probs, match_minute)

# Run simulation
simulate_match(final_probs)

