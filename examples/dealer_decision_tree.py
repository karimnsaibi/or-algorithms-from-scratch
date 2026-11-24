"""Sequential decision: when should an art dealer buy a painting?

Day 1 price $10,000. Each day the painting has a 60 % chance of being sold to someone
else. If still available, day 2 price is $20,000 (resale value), day 3 offers $24,000.

    python -m examples.dealer_decision_tree
"""
from or_algorithms import DecisionNode, ProbabilityNode, value_of_information


def build_tree():
    day3 = (ProbabilityNode('Day 3')
            .add_outcome(0.6, 0, 'Sold to someone else')
            .add_outcome(0.4, 24000, 'Still available: buy on day 3'))
    day2 = (DecisionNode('Day 2')
            .add_option('Buy on day 2', 20000)
            .add_option('Wait one more day', day3))
    day1 = (ProbabilityNode('Day 1 -> 2')
            .add_outcome(0.6, 0, 'Sold to someone else')
            .add_outcome(0.4, day2, 'Still available'))
    return (DecisionNode('Dealer')
            .add_option('Buy on day 1', 10000)
            .add_option('Wait one day', day1))


def main():
    tree = build_tree()
    tree.display()
    print('\nOptimal policy:', ' | '.join(tree.best_path()))

    # Perfect information: knowing in advance whether the painting survives to day 3 (p = 0.4^2).
    with_info = (ProbabilityNode('With perfect information')
                 .add_outcome(0.16, 24000, 'Survives to day 3: wait')
                 .add_outcome(0.84, 10000, 'Will be sold: buy on day 1'))
    print(f'\nExpected value without information: ${tree.evaluate():,.0f}')
    print(f'Expected value with perfect information: ${with_info.evaluate():,.0f}')
    print(f'EVPI (most the dealer should pay for the information): '
          f'${value_of_information(tree, with_info):,.0f}')


if __name__ == '__main__':
    main()
