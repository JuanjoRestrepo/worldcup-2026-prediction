"""Monte Carlo simulation for football matches and tournaments."""

from typing import Any

import numpy as np


class MonteCarloSimulator:
    def __init__(self, n_simulations: int = 10000, random_seed: int | None = 42):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)

    def simulate_match(self, home_xg: float, away_xg: float) -> dict[str, Any]:
        """
        Simulate a match n_simulations times introducing xG shocks for variance.
        """
        # Base xG arrays
        base_home = np.full(self.n_simulations, float(home_xg))
        base_away = np.full(self.n_simulations, float(away_xg))

        # 1. Penalties: ~15% chance per match to award +0.79 xG to a team
        home_penalties = self.rng.binomial(1, 0.075, self.n_simulations)
        away_penalties = self.rng.binomial(1, 0.075, self.n_simulations)

        # 2. Red Cards: ~10% chance per match.
        home_reds = self.rng.binomial(1, 0.05, self.n_simulations)
        away_reds = self.rng.binomial(1, 0.05, self.n_simulations)

        # 3. Random Gaussian Noise (Referee errors / form):
        home_noise = self.rng.normal(0, 0.1, self.n_simulations)
        away_noise = self.rng.normal(0, 0.1, self.n_simulations)

        # Calculate shocked xG
        home_shocked_xg = base_home + (home_penalties * 0.79) + home_noise
        away_shocked_xg = base_away + (away_penalties * 0.79) + away_noise

        # Apply red card multipliers
        home_multiplier = np.ones(self.n_simulations)
        away_multiplier = np.ones(self.n_simulations)

        home_multiplier[home_reds == 1] *= 0.7
        away_multiplier[home_reds == 1] *= 1.2

        home_multiplier[away_reds == 1] *= 1.2
        away_multiplier[away_reds == 1] *= 0.7

        home_final_xg = np.maximum(0.01, home_shocked_xg * home_multiplier)
        away_final_xg = np.maximum(0.01, away_shocked_xg * away_multiplier)

        # Draw from Poisson
        home_goals = self.rng.poisson(home_final_xg)
        away_goals = self.rng.poisson(away_final_xg)

        # Calculate outcomes
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)

        # Calculate score distributions
        scores: dict[str, int] = {}
        for h, a in zip(home_goals, away_goals):
            score = f"{h}-{a}"
            scores[score] = scores.get(score, 0) + 1

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_scores = {k: v / self.n_simulations for k, v in sorted_scores[:10]}

        return {
            "home_win_prob": float(home_wins / self.n_simulations),
            "draw_prob": float(draws / self.n_simulations),
            "away_win_prob": float(away_wins / self.n_simulations),
            "top_exact_scores": top_scores,
            "expected_home_goals": float(np.mean(home_goals)),
            "expected_away_goals": float(np.mean(away_goals)),
        }

    def simulate_knockout_tie(self, home_xg: float, away_xg: float) -> dict[str, float]:
        """
        Simulate a knockout match where draws are resolved by extra time and penalties.
        Returns the probability of each team advancing.
        """
        match_stats = self.simulate_match(home_xg, away_xg)

        hw = match_stats["home_win_prob"]
        dr = match_stats["draw_prob"]
        aw = match_stats["away_win_prob"]

        total_xg = home_xg + away_xg
        if total_xg == 0:
            total_xg = 0.0001

        home_et_adv = home_xg / total_xg
        away_et_adv = away_xg / total_xg

        # Assume in ET, 40% of matches are decided, 60% go to penalties
        et_decided = 0.4 * dr
        et_home_win = et_decided * home_et_adv
        et_away_win = et_decided * away_et_adv

        pens_prob = 0.6 * dr
        pens_home_win = pens_prob * 0.5
        pens_away_win = pens_prob * 0.5

        home_advances = hw + et_home_win + pens_home_win
        away_advances = aw + et_away_win + pens_away_win

        # Normalize just in case of floating point errors
        total = home_advances + away_advances

        return {
            "home_advances": home_advances / total,
            "away_advances": away_advances / total,
        }

    def simulate_bracket(
        self,
        match_probs: dict[str, dict[str, float]],
        initial_matchups: list[tuple[str, str]],
    ) -> dict[str, dict[str, float]]:
        """
        Simulates a knockout bracket.
        `match_probs` is a nested dict: match_probs[teamA][teamB] = probability teamA beats teamB in a knockout tie.
        `initial_matchups` is a list of tuples like [("France", "Germany"), ("Brazil", "Argentina"), ...] representing the Round of 16 or Quarterfinals, in order.

        Returns a dict of probabilities for each team reaching each stage.
        """
        stages = ["Round_of_16", "Quarter_Finals", "Semi_Finals", "Final", "Winner"]

        # Calculate number of stages based on number of initial matchups
        # 8 matchups -> Round of 16 (stage 0), QF (stage 1), SF (stage 2), Final (stage 3), Winner (stage 4)
        if len(initial_matchups) == 16:
            stages = [
                "Round_of_32",
                "Round_of_16",
                "Quarter_Finals",
                "Semi_Finals",
                "Final",
                "Winner",
            ]
        elif len(initial_matchups) == 8:
            stages = ["Round_of_16", "Quarter_Finals", "Semi_Finals", "Final", "Winner"]
        elif len(initial_matchups) == 4:
            stages = ["Quarter_Finals", "Semi_Finals", "Final", "Winner"]
        elif len(initial_matchups) == 2:
            stages = ["Semi_Finals", "Final", "Winner"]
        elif len(initial_matchups) == 1:
            stages = ["Final", "Winner"]

        results: dict[str, dict[str, float]] = {}
        for h, a in initial_matchups:
            results[h] = dict.fromkeys(stages, 0.0)
            results[a] = dict.fromkeys(stages, 0.0)
            results[h][stages[0]] = 1.0
            results[a][stages[0]] = 1.0

        # We will use Monte Carlo to simulate the bracket self.n_simulations times.
        # Alternatively, we could compute exact probabilities analytically, but Monte Carlo allows correlated events later.

        for sim in range(self.n_simulations):
            current_round = initial_matchups.copy()
            for stage_idx in range(len(stages) - 1):
                next_round = []
                winners = []

                for h, a in current_round:
                    # Lookup probability h beats a
                    prob_h_wins = match_probs.get(h, {}).get(a, 0.5)

                    if self.rng.random() < prob_h_wins:
                        winners.append(h)
                    else:
                        winners.append(a)

                    results[winners[-1]][stages[stage_idx + 1]] += 1.0

                # Pair up winners for next round
                for i in range(0, len(winners), 2):
                    if i + 1 < len(winners):
                        next_round.append((winners[i], winners[i + 1]))

                current_round = next_round

        # Normalize
        for team in results:
            for stage in stages:
                if stage != stages[0]:  # Stage 0 is always 1.0
                    results[team][stage] /= self.n_simulations

        return results
