'''Common functions and data structures used for defining trees.'''

from collections import namedtuple

Forest = namedtuple('Forest', 'trees')
Tree = namedtuple('Tree', 'feature splits children')
Split = namedtuple('Split', 'feature min max')
Leaf = namedtuple('Leaf', 'value')

INF = float('inf')


def print_tree(node, depth=0):
    """Print ascii tree / forest / leaf."""
    indent = ' ' * (depth * 2)
    if node.__class__ is Leaf:
        print('%s-> %s' % (indent, node.value))
    elif node.__class__ is Tree:
        for split, child in zip(node.splits, node.children):
            if child.__class__ is Leaf:
                print('%s(%s <= %s) -> %s' % (indent, split.feature, split.max, child.value))
            else:
                print('%s(%s <= %s)' % (indent, split.feature, split.max))
                print_tree(child, depth + 1)
    elif node.__class__ is Forest:
        for i, tree in enumerate(node.trees):
            print('%s# Tree %d / %d' % (indent, i, len(node.trees)))
            print_tree(tree, depth + 1)
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
