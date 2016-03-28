'''Common functions and data structures used for defining trees.'''

from collections import namedtuple
import numpy as np

Forest = namedtuple('Forest', 'trees')
Tree = namedtuple('Tree', 'feature splits children')
Split = namedtuple('Split', 'feature min max')
Leaf = namedtuple('Leaf', 'value')

INF = float('inf')


def print_tree(node, depth=0):
    """Print ascii tree / forest / leaf."""
    for line in string_tree(node, depth):
        print(line)


def string_tree(node, depth=0):
    """emit ascii tree / forest / leaf."""
    indent = ' ' * (depth * 2)
    if node.__class__ is Leaf:
        yield '%s-> %s' % (indent, node.value)
    elif node.__class__ is Tree:
        for split, child in zip(node.splits, node.children):
            if child.__class__ is Leaf:
                yield '%s(%s <= %s) -> %s' % (indent, split.feature, split.max, child.value)
            else:
                yield '%s(%s <= %s)' % (indent, split.feature, split.max)
                yield from string_tree(child, depth + 1)
    elif node.__class__ is Forest:
        for i, tree in enumerate(node.trees):
            yield '%s# Tree %d / %d' % (indent, i, len(node.trees))
            yield from string_tree(tree, depth + 1)
    else:
        raise ValueError('Unexpected Value: %s', node)


def make_tree(var, cutpoints, children):
    """Build tree tuple from cutpoints and children."""
    splits = []
    prev_high = -INF
    for point in cutpoints:
        splits.append(Split(var, prev_high, point))
        prev_high = point
    splits.append(Split(var, prev_high, INF))
    return Tree(var, splits, children)


def calculate_value(node, val_dict):
    """Calculate value returned by forest"""
    if node.__class__ is Leaf:
        return node.value
    elif node.__class__ is Tree:
        feature_val = val_dict[node.feature]
        for split, child in zip(node.splits, node.children):
            if split.min < feature_val <= split.max:
                return calculate_value(child, val_dict)
    elif node.__class__ is Forest:
        return np.mean([calculate_value(tree, val_dict) for tree in node.trees])
    raise NotImplementedError('class=%s' % node.__class__)
