from fractions import Fraction

# Function to convert fractional odds (A/B) to probability
def fractional_to_probability(odd):
    A, B = odd
    return B / (A + B)

# Function to convert decimal probability to fractional odds
def decimal_to_fractional_odds(probability_decimal):
    if probability_decimal == 1:
        return "1/0 (certainty)"
    if probability_decimal == 0:
        return "∞ (impossible)"
    
    # Calculate the odds as (probability / (1 - probability))
    odds = probability_decimal / (1 - probability_decimal)
    
    # Convert the odds to a fraction
    fractional_odds = Fraction(odds).limit_denominator(1000)
    
    return f"{fractional_odds.numerator}/{fractional_odds.denominator}"


# Function to calculate average probability from a list of fractional odds
def calculate_average_probability(odds_list):
    probabilities = [fractional_to_probability(odd) for odd in odds_list]
    return sum(probabilities) / len(probabilities) * 100  # Return in percentage

# Given odds for each team
odds = {
    'England': [(1, 300), (1, 500), (1, 100)],
    'Newcastle': [
    (4, 5), (5, 6), (17, 20), (8, 11), (4, 5), (17, 20),
    (4, 6), (8, 11), (8, 11), (8, 11), (4, 5), (4, 6),
    (8, 11), (8, 11), (8, 11), (4, 5), (4, 5), (8, 11),
    (5, 6), (4, 5), (8, 11), (8, 11), (8, 11), (1, 16), (1, 16), (1, 16), (1, 20), (1, 16), (1, 16),
    (1, 14), (1, 10), (1, 20), (1, 16), (1, 16),
    (1, 20), (1, 20), (1, 20), (1, 16), (1, 10),
    (1, 16), (2, 13), (1, 8)
],



    'Leeds': [
    (1, 200), (1, 100), (1, 100), (1, 1000), (1, 100), 
    (1, 33), (1, 40), (1, 250), (1, 100), (1, 100), 
    (1, 250), (1, 33), (1, 100), (1, 200), (1, 25)
]
}

# Calculate average probabilities for each team
average_probabilities = {team: calculate_average_probability(odds_list) for team, odds_list in odds.items()}

# Convert to decimals for multiplication
probabilities_decimal = {team: avg / 100 for team, avg in average_probabilities.items()}

# Multiply the decimal probabilities to get combined probability
overall_probability = probabilities_decimal['England'] * probabilities_decimal['Newcastle'] * probabilities_decimal['Leeds']
overall_probability_percent = overall_probability * 100  # Convert back to percentage

# Convert to fractional odds
overall_fractional_odds = decimal_to_fractional_odds(overall_probability)

# Print results
print("\nProbabilities and Odds:")
for team, avg_prob in average_probabilities.items():
    print(f"Average probability for {team} qualifying: {avg_prob:.4f}%")

print(f"\nOverall probability for all events: {overall_probability_percent:.5f}%")
print(f"Fractional odds for all events occurring: {overall_fractional_odds}")


