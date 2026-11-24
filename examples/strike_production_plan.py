"""Production planning under strike uncertainty: every classic decision criterion.

A factory can produce for 1, 2 or 3 weeks before a possible strike (none, 1 or 2 weeks).

    python -m examples.strike_production_plan
"""
import pandas as pd

from or_algorithms import decision_criteria as dc

PROFIT = pd.DataFrame(
    {'No strike': [225000, 195000, 145000],
     '1-week strike': [150000, 195000, 145000],
     '2-week strike': [75000, 120000, 145000]},
    index=['Produce 1 week', 'Produce 2 weeks', 'Produce 3 weeks'],
)
PROBABILITIES = [0.2, 0.4, 0.4]


def main():
    print('Profit table (TND):')
    print(PROFIT, '\n')
    print('Regret table:')
    print(dc.regret_table(PROFIT), '\n')
    print('Recommendation by criterion:')
    print(dc.summary(PROFIT, PROBABILITIES, alpha=0.7).to_string(), '\n')
    info = dc.evpi(PROFIT, PROBABILITIES)
    print(f"Expected value with perfect information: {info['ev_with_perfect_info']:,.0f} TND")
    print(f"Best expected value without it:          {info['best_emv']:,.0f} TND")
    print(f"EVPI:                                    {info['evpi']:,.0f} TND")


if __name__ == '__main__':
    main()
