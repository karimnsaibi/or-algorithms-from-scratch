"""Zenith Furniture Co.: weekly production mix (linear programming).

Desks earn $70, chairs $50. Weekly capacity: 240 units of oak, 100 carpentry hours,
60 finishing hours. How many of each should be produced?

    python -m examples.zenith_production_mix            # solution + shadow prices
    python -m examples.zenith_production_mix --steps    # every simplex tableau
    python -m examples.zenith_production_mix --plot     # graphical method -> docs/zenith_graphical.png
"""
import sys
from pathlib import Path

import numpy as np

from or_algorithms import SimplexSolver, format_tableau

PRODUCTS = ['Executive desks', 'Ergonomic chairs']
RESOURCES = ['Oak wood (units)', 'Carpentry (hours)', 'Finishing (hours)']
PROFIT = [70, 50]
USAGE = [[4, 3], [2, 1], [1, 1]]
CAPACITY = [240, 100, 60]


def main():
    solver = SimplexSolver(PROFIT, USAGE, CAPACITY, maximize=True, trace='--steps' in sys.argv)
    result = solver.solve()

    if '--steps' in sys.argv:
        names = ['x1', 'x2', 's1', 's2', 's3']
        for k, snap in enumerate(result.history):
            print(f'\nTableau {k}')
            print(format_tableau(snap, names))

    print('\n' + result.summary(PRODUCTS, RESOURCES))
    print('\nReading the shadow prices:')
    for name, price, slack in zip(RESOURCES, result.shadow_prices, result.slacks):
        if price > 0:
            print(f'  - One more unit of {name.split(" (")[0].lower()} adds ${price:,.0f} of weekly profit.')
        else:
            print(f'  - {name.split(" (")[0]} is not a bottleneck ({slack:,.0f} units unused).')

    if '--plot' in sys.argv:
        plot(result)


def plot(result):
    import matplotlib.pyplot as plt

    x = np.linspace(0, 70, 400)
    fig, ax = plt.subplots(figsize=(6, 5))
    for (a1, a2), cap, name in zip(USAGE, CAPACITY, RESOURCES):
        ax.plot(x, (cap - a1 * x) / a2, label=name)
    X, Y = np.meshgrid(x, np.linspace(0, 110, 400))
    feasible = np.all([a1 * X + a2 * Y <= cap for (a1, a2), cap in zip(USAGE, CAPACITY)], axis=0)
    ax.contourf(X, Y, feasible, levels=[0.5, 1], alpha=0.25)
    ax.plot(*result.x, 'o', color='black')
    ax.annotate(f'optimum {result.x[0]:.0f} desks, {result.x[1]:.0f} chairs\nprofit ${result.objective:,.0f}',
                result.x, xytext=(result.x[0] + 4, result.x[1] + 18), arrowprops={'arrowstyle': '->'})
    ax.set(xlim=(0, 70), ylim=(0, 110), xlabel='Desks per week', ylabel='Chairs per week',
           title='Zenith Furniture: feasible region')
    ax.legend(loc='upper right', fontsize=8)
    out = Path(__file__).resolve().parents[1] / 'docs' / 'zenith_graphical.png'
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
