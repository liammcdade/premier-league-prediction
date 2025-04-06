# utils.py
def get_average_probability(odds_str):
    """Convert odds to probability."""
    # Example: Convert "11/10" to probability (11/21)
    numerator, denominator = map(int, odds_str.split('/'))
    return numerator / denominator
