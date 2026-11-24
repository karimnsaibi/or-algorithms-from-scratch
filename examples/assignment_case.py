"""Assignment problem: give each technician one job at minimum total cost.

    python -m examples.assignment_case
"""
import pandas as pd

from or_algorithms import hungarian, reduce_matrix

COST = pd.DataFrame(
    [[150, 200, 125],
     [280, 150, 180],
     [300, 220, 340]],
    index=['Technician A', 'Technician B', 'Technician C'],
    columns=['Job 1', 'Job 2', 'Job 3'],
)


def main():
    print('Cost matrix (TND):')
    print(COST, '\n')

    after_rows, after_cols = reduce_matrix(COST.values)
    print('Step 1: subtract each row minimum')
    print(pd.DataFrame(after_rows, index=COST.index, columns=COST.columns), '\n')
    print('Step 2: subtract each column minimum')
    print(pd.DataFrame(after_cols, index=COST.index, columns=COST.columns), '\n')

    rows, cols, total = hungarian(COST.values)
    print('Optimal assignment:')
    for r, c in zip(rows, cols):
        print(f'  {COST.index[r]} -> {COST.columns[c]}  ({COST.iat[r, c]} TND)')
    print(f'Total cost: {total:,.0f} TND')


if __name__ == '__main__':
    main()
