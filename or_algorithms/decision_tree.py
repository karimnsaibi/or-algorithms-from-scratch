"""Decision trees solved by backward induction (rollback).

A tree is built from two node types:
- DecisionNode: the decision maker picks the option with the best value.
- ProbabilityNode (chance node): its value is the probability-weighted average of its outcomes.
Leaves are plain numbers (payoffs).
"""
from __future__ import annotations

PROB_TOLERANCE = 1e-9


class Node:
    """Base class for all nodes."""

    def __init__(self, name: str):
        self.name = name

    def evaluate(self) -> float:
        raise NotImplementedError

    def display(self, prefix: str = '') -> None:
        print(self.render(prefix))

    def render(self, prefix: str = '') -> str:
        raise NotImplementedError


def _value(child) -> float:
    return child.evaluate() if isinstance(child, Node) else float(child)


def _render_child(child, prefix: str) -> str:
    if isinstance(child, Node):
        return child.render(prefix)
    return f'{prefix}└── Payoff: {child:,}'


class DecisionNode(Node):
    """Chooses the option with the highest (or lowest, for costs) value."""

    def __init__(self, name: str, minimize: bool = False):
        super().__init__(name)
        self.options = {}            # option name -> Node or payoff
        self.minimize = minimize
        self.best_option = None
        self.best_value = None

    def add_option(self, option_name: str, next_node) -> 'DecisionNode':
        self.options[option_name] = next_node
        return self

    def evaluate(self) -> float:
        if not self.options:
            raise ValueError(f"Decision node '{self.name}' has no options")
        values = {name: _value(child) for name, child in self.options.items()}
        pick = min if self.minimize else max
        self.best_option = pick(values, key=values.get)
        self.best_value = values[self.best_option]
        return self.best_value

    def best_path(self) -> list[str]:
        """Sequence of optimal choices, following the best option downwards."""
        self.evaluate()
        path = [f'{self.name}: {self.best_option}']
        child = self.options[self.best_option]
        while isinstance(child, ProbabilityNode) and len(child.outcomes) == 1:
            child = child.outcomes[0][1]
        if isinstance(child, DecisionNode):
            path += child.best_path()
        return path

    def render(self, prefix: str = '') -> str:
        value = self.evaluate()
        lines = [f'{prefix}[Decision: {self.name}] best = {self.best_option} -> {value:,.2f}']
        for name, child in self.options.items():
            mark = '*' if name == self.best_option else ' '
            lines.append(f'{prefix} ├─{mark} {name}')
            lines.append(_render_child(child, prefix + ' │   '))
        return '\n'.join(lines)


class ProbabilityNode(Node):
    """Chance node: expected value of its outcomes."""

    def __init__(self, name: str):
        super().__init__(name)
        self.outcomes = []           # (probability, Node or payoff, label)
        self.expected = None

    def add_outcome(self, probability: float, outcome, name: str | None = None) -> 'ProbabilityNode':
        if not 0 <= probability <= 1:
            raise ValueError('Probability must be between 0 and 1.')
        self.outcomes.append((probability, outcome, name or f'outcome {len(self.outcomes) + 1}'))
        return self

    def evaluate(self) -> float:
        total_p = sum(p for p, _, _ in self.outcomes)
        if abs(total_p - 1) > PROB_TOLERANCE:
            raise ValueError(f"Probabilities at '{self.name}' sum to {total_p}, not 1")
        self.expected = sum(p * _value(child) for p, child, _ in self.outcomes)
        return self.expected

    def render(self, prefix: str = '') -> str:
        value = self.evaluate()
        lines = [f'{prefix}[Chance: {self.name}] expected value = {value:,.2f}']
        for p, child, label in self.outcomes:
            lines.append(f'{prefix} ├── {label} (p={p:g})')
            lines.append(_render_child(child, prefix + ' │   '))
        return '\n'.join(lines)


ChanceNode = ProbabilityNode


def value_of_information(without_info: Node, with_info: Node) -> float:
    """EVPI / EVSI: expected value with the information minus without it."""
    return with_info.evaluate() - without_info.evaluate()
