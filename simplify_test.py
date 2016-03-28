'''
Test simplifiy functions match original forest.
'''

import simplify_greedy_runtime

from onetree_common_test import make_small_test_forest
from onetree_common import calculate_value


def check_equal_calculations(node1, node2, input_dicts, delta=0.00001):
    """Test both models produce equivalent reuslts."""
    for entry in input_dicts:
        assert abs(calculate_value(node1, entry) - calculate_value(node2, entry)) < delta


def test_greedy_runtime():
    original = make_small_test_forest()
    simplified = simplify_greedy_runtime.simplify(original)
    stratified = [-1, -.5, 0, .21, .5, .51, .76, 1.0]
    input_dicts = [{'X_1': x1, 'X_2': x2, 'X_3': x3}
                   for x1 in stratified
                   for x2 in stratified
                   for x3 in stratified]
    check_equal_calculations(original, simplified, input_dicts)
