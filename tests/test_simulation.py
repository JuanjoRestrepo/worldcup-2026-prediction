from backend.modeling.simulation import MonteCarloSimulator


def test_simulate_match():
    simulator = MonteCarloSimulator(n_simulations=1000)
    stats = simulator.simulate_match(home_xg=1.5, away_xg=1.0)

    assert "home_win_prob" in stats
    assert "draw_prob" in stats
    assert "away_win_prob" in stats
    assert "top_exact_scores" in stats

    # Probabilities should roughly sum to 1.0
    total_prob = stats["home_win_prob"] + stats["draw_prob"] + stats["away_win_prob"]
    assert 0.99 <= total_prob <= 1.01


def test_simulate_knockout_tie():
    simulator = MonteCarloSimulator(n_simulations=1000)
    probs = simulator.simulate_knockout_tie(home_xg=2.0, away_xg=0.5)

    assert "home_advances" in probs
    assert "away_advances" in probs

    # Probabilities should sum to 1.0
    total_prob = probs["home_advances"] + probs["away_advances"]
    assert 0.99 <= total_prob <= 1.01

    # Strong favorite should have higher probability of advancing
    assert probs["home_advances"] > probs["away_advances"]


def test_simulate_bracket():
    simulator = MonteCarloSimulator(n_simulations=1000)

    match_probs = {
        "A": {"C": 0.7, "D": 0.9, "B": 0.8},
        "B": {"C": 0.5, "D": 0.7, "A": 0.2},
        "C": {"A": 0.3, "B": 0.5, "D": 0.6},
        "D": {"A": 0.1, "B": 0.3, "C": 0.4},
    }

    matchups = [("A", "B"), ("C", "D")]

    results = simulator.simulate_bracket(match_probs, matchups)

    # A has highest prob
    assert results["A"]["Semi_Finals"] == 1.0
    assert results["A"]["Winner"] > 0.4
