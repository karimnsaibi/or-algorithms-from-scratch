"""Decision making under uncertainty and risk on a payoff matrix.

The payoff table is a pandas DataFrame: rows = decisions (alternatives),
columns = states of nature, values = payoffs (profits; use cost=True for costs).
Every criterion returns (best_decision, scores_per_decision).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sign(cost: bool) -> int:
    return -1 if cost else 1


def _best(scores: pd.Series, cost: bool):
    """Lowest score wins for costs, highest for profits."""
    return (scores.idxmin() if cost else scores.idxmax()), scores


def maximax(payoffs: pd.DataFrame, cost: bool = False):
    """Optimist: best of the best outcomes."""
    scores = payoffs.min(axis=1) if cost else payoffs.max(axis=1)
    return _best(scores, cost)


def maximin(payoffs: pd.DataFrame, cost: bool = False):
    """Pessimist (Wald): best of the worst outcomes."""
    scores = payoffs.max(axis=1) if cost else payoffs.min(axis=1)
    return _best(scores, cost)


def hurwicz(payoffs: pd.DataFrame, alpha: float, cost: bool = False):
    """Weighted optimism: alpha * best outcome + (1 - alpha) * worst outcome."""
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be between 0 and 1')
    best = payoffs.min(axis=1) if cost else payoffs.max(axis=1)
    worst = payoffs.max(axis=1) if cost else payoffs.min(axis=1)
    return _best(alpha * best + (1 - alpha) * worst, cost)


def laplace(payoffs: pd.DataFrame, cost: bool = False):
    """Equally likely states: plain average."""
    return _best(payoffs.mean(axis=1), cost)


def regret_table(payoffs: pd.DataFrame, cost: bool = False) -> pd.DataFrame:
    """Opportunity loss of each decision in each state."""
    if cost:
        return payoffs - payoffs.min(axis=0)
    return payoffs.max(axis=0) - payoffs


def minimax_regret(payoffs: pd.DataFrame, cost: bool = False):
    """Savage: smallest maximum regret."""
    scores = regret_table(payoffs, cost).max(axis=1)
    return scores.idxmin(), scores


def expected_value(payoffs: pd.DataFrame, probabilities, cost: bool = False):
    """Decision under risk: expected monetary value (EMV) of each decision."""
    p = _check_probabilities(payoffs, probabilities)
    return _best(payoffs @ p, cost)


def evpi(payoffs: pd.DataFrame, probabilities, cost: bool = False) -> dict:
    """Expected value of perfect information = EV with perfect information - best EMV."""
    p = _check_probabilities(payoffs, probabilities)
    best_per_state = payoffs.min(axis=0) if cost else payoffs.max(axis=0)
    ev_wpi = float(best_per_state @ p)
    _, emv = expected_value(payoffs, p, cost)
    best_emv = float(emv.min() if cost else emv.max())
    return {'ev_with_perfect_info': ev_wpi, 'best_emv': best_emv,
            'evpi': _sign(cost) * (ev_wpi - best_emv)}


def summary(payoffs: pd.DataFrame, probabilities=None, alpha: float = 0.5, cost: bool = False) -> pd.DataFrame:
    """One row per criterion with the recommended decision and its score."""
    rows = {
        'Maximax (optimistic)': maximax(payoffs, cost),
        'Maximin (pessimistic)': maximin(payoffs, cost),
        f'Hurwicz (alpha={alpha:g})': hurwicz(payoffs, alpha, cost),
        'Laplace (equal likelihood)': laplace(payoffs, cost),
        'Minimax regret (Savage)': minimax_regret(payoffs, cost),
    }
    if probabilities is not None:
        rows['Expected value (EMV)'] = expected_value(payoffs, probabilities, cost)
    return pd.DataFrame({'decision': {k: v[0] for k, v in rows.items()},
                         'score': {k: float(v[1][v[0]]) for k, v in rows.items()}})


def _check_probabilities(payoffs: pd.DataFrame, probabilities) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (payoffs.shape[1],):
        raise ValueError('one probability per state (column) is required')
    if np.any(p < 0) or abs(p.sum() - 1) > 1e-9:
        raise ValueError('probabilities must be non-negative and sum to 1')
    return p
