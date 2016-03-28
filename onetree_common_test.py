"""Test module for onetree_common.py"""

import numpy as np

from onetree_common import Forest, Leaf
from onetree_common import make_tree, string_tree, calculate_value


def make_small_test_forest():
    """Build a small forest for testing."""
    # Sample forest, roughtly as a result of predicting 'Y = X_1'.
    _tree1 = make_tree('X_1', [.5], [Leaf(0.25), Leaf(0.75)])
    _tree2 = make_tree('X_1', [.5], [Leaf(0.30), Leaf(0.90)])
    _tree3 = make_tree('X_2', [.5], [
        make_tree('X_3', [.5], [Leaf(0.35), Leaf(0.75)]),
        Leaf(0.45)])
    _tree4 = Leaf(0.45)
    _tree5 = make_tree('X_3', [.5], [Leaf(0.30), Leaf(0.90)])
    _tree6 = make_tree('X_3', [.4, .6], [_tree1, _tree2, _tree3])
    _tree7 = make_tree('X_3', [.4, .6], [_tree6, _tree1, _tree6])
    _forest = Forest([_tree1,
                      _tree2,
                      _tree3,
                      _tree4,
                      _tree5,
                      _tree5,
                      _tree6,
                      _tree7,
                      make_tree('X_1', [.2, .3], [Leaf(.1), Leaf(.25), Leaf(.65)])])
    return _forest


def test_calculate_value():
    leaf = Leaf(0.45)
    assert calculate_value(leaf, {'X_1': 7}) == 0.45

    # Depth 1 tree
    tree_1 = make_tree('X_1', [.5], [Leaf(0.25), Leaf(0.75)])
    assert calculate_value(tree_1, {'X_1': -10}) == 0.25
    assert calculate_value(tree_1, {'X_1': .5}) == 0.25
    assert calculate_value(tree_1, {'X_1': 7}) == 0.75

    # Depth 2 tree
    tree_2 = make_tree('X_1', [.5], [Leaf(0.25),
                                     make_tree('X_2', [.9], [Leaf(0.35), Leaf(0.85)])])
    assert calculate_value(tree_2, {'X_1': -10}) == 0.25
    assert calculate_value(tree_2, {'X_1': .5}) == 0.25
    assert calculate_value(tree_2, {'X_1': 7, 'X_2': 0.1}) == 0.35
    assert calculate_value(tree_2, {'X_1': 7, 'X_2': 0.9}) == 0.35
    assert calculate_value(tree_2, {'X_1': 7, 'X_2': 77}) == 0.85

    try:
        calculate_value(tree_2, {'X_1': 5})
        assert 'Expected exception'
    except KeyError:
        pass

    # Non-binary tree.
    ternary = make_tree('X_3', [.4, .6], [Leaf(0.1), Leaf(0.2), Leaf(0.3)])
    assert calculate_value(ternary, {'X_3': -1}) == 0.1
    assert calculate_value(ternary, {'X_3': 0.5}) == 0.2
    assert calculate_value(ternary, {'X_3': 1.0}) == 0.3

    # Forest.
    forest = make_small_test_forest()
    assert calculate_value(forest, {'X_1': 0, 'X_2': 1, 'X_3': 0}) == np.mean([0.25, 0.3, 0.45, 0.45, 0.3, 0.3, 0.25, 0.25, 0.1])
    print('\nAll tests pass 2.')


def test_string_tree():
    forest = make_small_test_forest()

    tiny_forest = Forest(forest.trees[:3])
    tiny_forest_string = '\n'.join(string_tree(tiny_forest))

    assert tiny_forest_string == """\
# Tree 0 / 3
  (X_1 <= 0.5) -> 0.25
  (X_1 <= inf) -> 0.75
# Tree 1 / 3
  (X_1 <= 0.5) -> 0.3
  (X_1 <= inf) -> 0.9
# Tree 2 / 3
  (X_2 <= 0.5)
    (X_3 <= 0.5) -> 0.35
    (X_3 <= inf) -> 0.75
  (X_2 <= inf) -> 0.45"""
