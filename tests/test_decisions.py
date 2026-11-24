import pandas as pd
import pytest

from or_algorithms import DecisionNode, ProbabilityNode, value_of_information
from or_algorithms import decision_criteria as dc


# ----------------------------------------------------------- decision trees
def dealer_tree():
    """Sequential painting-dealer problem (Decision & Game Theory, tutorial 2)."""
    day3 = ProbabilityNode('Day 3').add_outcome(0.6, 0, 'Sold').add_outcome(0.4, 24000, 'Buy on day 3')
    day2 = DecisionNode('Day 2').add_option('Buy on day 2', 20000).add_option('Wait 1 day', day3)
    day1 = ProbabilityNode('Day 1').add_outcome(0.6, 0, 'Sold').add_outcome(0.4, day2, 'Still available')
    return DecisionNode('Dealer').add_option('Buy on day 1', 10000).add_option('Wait 1 day', day1)


def test_dealer_rollback():
    tree = dealer_tree()
    assert tree.evaluate() == pytest.approx(10000)
    assert tree.best_option == 'Buy on day 1'
    assert tree.options['Wait 1 day'].evaluate() == pytest.approx(8000)


def test_dealer_evpi():
    # With perfect information: painting still there on day 3 with p = 0.4^2 = 0.16
    with_info = (ProbabilityNode('Perfect information')
                 .add_outcome(0.16, 24000, 'Available on day 3')
                 .add_outcome(0.84, 10000, 'Sold before day 3'))
    assert value_of_information(dealer_tree(), with_info) == pytest.approx(2240)


def test_probabilities_must_sum_to_one():
    node = ProbabilityNode('bad').add_outcome(0.5, 1).add_outcome(0.4, 2)
    with pytest.raises(ValueError):
        node.evaluate()


def test_cost_minimizing_decision():
    node = DecisionNode('Supplier', minimize=True).add_option('A', 120).add_option('B', 95)
    assert node.evaluate() == 95 and node.best_option == 'B'


def test_render_marks_best_option():
    text = dealer_tree().render()
    assert '* Buy on day 1' in text


# ------------------------------------------------------- decision criteria
@pytest.fixture
def strike():
    """Production plan under strike uncertainty (Decision & Game Theory, question 5)."""
    return pd.DataFrame(
        {'No strike': [225000, 195000, 145000],
         '1-week strike': [150000, 195000, 145000],
         '2-week strike': [75000, 120000, 145000]},
        index=['Produce 1 week', 'Produce 2 weeks', 'Produce 3 weeks'])


def test_expected_value_and_evpi(strike):
    best, emv = dc.expected_value(strike, [0.2, 0.4, 0.4])
    assert best == 'Produce 2 weeks' and emv[best] == pytest.approx(165000)
    info = dc.evpi(strike, [0.2, 0.4, 0.4])
    assert info['ev_with_perfect_info'] == pytest.approx(181000)
    assert info['evpi'] == pytest.approx(16000)


def test_classic_criteria(strike):
    assert dc.maximax(strike)[0] == 'Produce 1 week'
    assert dc.maximin(strike)[0] == 'Produce 3 weeks'
    assert dc.laplace(strike)[0] == 'Produce 2 weeks'
    best, scores = dc.minimax_regret(strike)
    assert best == 'Produce 2 weeks' and scores[best] == pytest.approx(30000)
    assert dc.hurwicz(strike, alpha=1)[0] == dc.maximax(strike)[0]
    assert dc.hurwicz(strike, alpha=0)[0] == dc.maximin(strike)[0]


def test_cost_tables_flip_the_direction():
    costs = pd.DataFrame({'low': [10, 20], 'high': [50, 30]}, index=['A', 'B'])
    assert dc.maximin(costs, cost=True)[0] == 'B'        # smallest worst-case cost
    assert dc.maximax(costs, cost=True)[0] == 'A'        # smallest best-case cost
    assert dc.evpi(costs, [0.5, 0.5], cost=True)['evpi'] == pytest.approx(5)


def test_summary_table(strike):
    table = dc.summary(strike, [0.2, 0.4, 0.4], alpha=0.6)
    assert len(table) == 6 and 'Expected value (EMV)' in table.index
