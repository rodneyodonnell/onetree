"""Script to convert a decision forest into a single tree.

## Implemented approach

The initial approach implemented here is:
 1. Find the feature, F, which is most most frequently split on in a random
    path through the forest.
 2. Find the set of all split points on F in the forest (one point for a binary
    variable, possibly many for continuous, etc.)
 3. Transform the forest into a tree root which splits on 'F', and a child
    forest under each branch.
 4. In each child forest, remove each splits on 'F' and  replace it with
    the appropriate child node matching the split choice.
 5. Repeat recursively for each forest in the new tree until no forests remain.

This approach also allows us to prune the tree by bucketing common tree values.
This is particulaly useful for classifiers where we only care about true/false
and can drastically reduce the tree size.

In theory this should make it much faster to evaluate a tree.
Instead of 100 trees of depth 10, we might have a single tree of depth 20 giving
a theoretical ~50x speedup.  This is so far untested.

In practice, to fully colapse a tree using this algorithm we need to explore
all potential branches in the tree, worst case this may be the product of
the number of split points per variable (i.e., very expensive).


## Alternative approach

An alternative (not implemented yet) is to modify step 1 to instead find a
split point which results in the minumum growth in tree size.  E.g., If there
is a binary variable B which is split on every path through every tree in the
forest we could split run step 3 splitting on B and have two be left with two
forests who's total size is equal to the original forest size.  Evaluating
this tree would be faster as we only need to split on B once, instead of once
for every tree in the forest.  This is the ideal case for extracting a variable
from a forest.

Splitting on binary variables not used in every path would cause growth in the
forest proportional to how often it is used.  So if B was used in 70% of the
forest paths, after the split we'd have two forests, each of (.7*.5+.3=) 65% of
the original forest size (so a 30% growth in forest size).

Splitting on continuous variables works too, but can expand the tree more
quickly.  If we split on B < 10, then all splits on B in the left branch
can be collapsed if the split point <= 10, and all spits on the right
branch can be collapsed if the split poinxbxbt >= 10.  So when performing a binary
split on a continuous B (where B is used in every forest path) the size of
each child forest is reduced by 25%, so the final forest size increases by 50%.


NOTE: This code is mostly untested and has only been used on toy problems so far
      (see __main__).  It has the potential to scale to much larger networks,
      but could hit combinaitonal issues based on forest topology, etc.
"""

from collections import Counter
from collections import defaultdict
from collections import namedtuple
import math

Forest = namedtuple('Forest', 'trees')
Tree = namedtuple('Tree', 'feature splits children')
Split = namedtuple('Split', 'feature min max')
WeightedSplit = namedtuple('WeightedSplit', 'weight split')
Leaf = namedtuple('Leaf', 'value')


def make_tree(var, cutpoints, children):
    """Build tree tuple from cutpoints and children."""
    inf = float('Inf')
    splits = []
    prev_high = -inf
    for point in cutpoints:
        splits.append(Split(var, prev_high, point))
        prev_high = point
    splits.append(Split(var, prev_high, inf))
    return Tree(var, splits, children)


def print_tree(node, depth=0):
    """Print ascii tree / forest."""
    indent = ' ' * (depth * 2)
    if node.__class__ is Leaf:
        print '%s-> %s' % (indent, node.value)
    elif node.__class__ is Tree:
        for split, child in zip(node.splits, node.children):
            if child.__class__ is Leaf:
                print '%s(%s < %s) -> %s' % (indent, split.feature, split.max, child.value)
            else:
                print '%s(%s < %s)' % (indent, split.feature, split.max)
                print_tree(child, depth + 1)
    elif node.__class__ is Forest:
        for i, tree in enumerate(node.trees):
            print '%s# Tree %d / %d' % (indent, i, len(node.trees))
            print_tree(tree, depth + 1)
    else:
        raise ValueError('Unexpected Value: %s', node)


def get_weighted_splits(node, weight=1.0):
    """Find all splits and how many times each will be split on per element."""
    if node.__class__ is Leaf:
        return
    elif node.__class__ is Tree:
        for split in node.splits:
            yield WeightedSplit(weight, split)
        for child in node.children:
            # Naively assume equal portion of elements will be assigned to each child.
            for weighted_split in get_weighted_splits(child, weight / len(node.children)):
                yield weighted_split
    elif node.__class__ is Forest:
        for tree in node.trees:
            # To classify a single element, it must be classified by each tree in a
            # forest, so assign full weight to each.
            for weighted_split in get_weighted_splits(tree, weight):
                yield weighted_split
    else:
        raise ValueError('Unexpected Value: %s', node)


def find_best_split(weighted_splits):
    """Of all features used in a forest, find the best to frontload in a tree."""
    # Expected number of times we will split on feature X per element classified.
    weights = Counter()
    # Set of all split points for feature X.
    cutpoints = defaultdict(set)
    for weight, split in weighted_splits:
        weights[split.feature] += weight
        cutpoints[split.feature].add(split.min)
        cutpoints[split.feature].add(split.max)
    # Best split has maximum average savings. By splitting on feature X up front:
    # - We save weight[feat] split operations on average in the forest.
    # - We use a ~log_2(#cutpoints) binary search to decide which forest to use.

    def saving(feat):
        return weights[feat] - math.log(len(cutpoints[feat]), 2)
    best = max((feat for feat in cutpoints), key=saving)

    sorted_cuts = sorted(cutpoints[best])
    return [Split(best, low, high)
            for low, high in zip(sorted_cuts, sorted_cuts[1:])]


def filter_by_split(node, split):
    """Strip out all brances not consistent with 'split'."""
    assert split.__class__ is Split
    if node.__class__ is Leaf:
        return node
    elif node.__class__ is Tree:
        if node.feature == split.feature:
            # Find the child which matches the split filter.
            for c_split, child in zip(node.splits, node.children):
                if c_split.min <= split.min and c_split.max >= split.max:
                    return child
            raise ValueError('"split" should always fit inside a single node.split')
        else:  # node.feature != split.feature
            return Tree(node.feature, node.splits, [filter_by_split(child, split) for child in node.children])
    elif node.__class__ is Forest:
        return Forest([filter_by_split(tree, split) for tree in node.trees])
    else:
        raise ValueError('Unexpected Value: %s', node)


def collapse_leaf_forest(forest):
    """If forest is nothing but leaves, repalce with single leaf."""
    return Leaf(sum(leaf.value for leaf in forest.trees) / len(forest.trees))


def merge_buckets(splits, children, bucketize_fn):
    """Merge buckets together when bucketize_fn(Node) is equal.

    This is useful to reduce to tree size, particularly for classification
    tasks where you only care about a small set of values (true/false, etc.)
    """
    if bucketize_fn is None:
        return splits, children

    ret_splits, ret_children = [], []
    for split, child in zip(splits, children):
        if child.__class__ is Leaf:
            child = Leaf(bucketize_fn(child.value))
        # Merge buckets with previous is both are equal.
        if ret_children and ret_children[-1] == child:
            assert ret_splits[-1].feature == split.feature
            ret_splits[-1] = ret_splits[-1]._replace(max=split.max)
        else:
            ret_splits.append(split)
            ret_children.append(child)

    assert len(ret_splits) == len(ret_children)
    assert len(ret_splits) > 0
    return ret_splits, ret_children


def simplify(forest, bucketize_fn=None):
    """Recursively simplify forest into a single tree."""
    weighted_splits = list(get_weighted_splits(forest))

    # Nothing so split on, so we know only leaves remain.
    if not weighted_splits:
        return collapse_leaf_forest(forest)

    # Turn forest into a tree, where each child is a forest.
    splits = find_best_split(weighted_splits)
    children = [simplify(filter_by_split(forest, split), bucketize_fn)
                for split in splits]

    splits, children = merge_buckets(splits, children, bucketize_fn)
    if len(splits) > 1:
        return Tree(splits[0].feature, splits, children)
    if len(splits) == 1:
        return children[0]


if __name__ == '__main__':
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

    print '\n## Full Tree'
    print_tree(_forest)

    print '\n## Simplified'
    print_tree(simplify(_forest))

    print '\n## Simplified with buckets'
    bucket_size = 0.1
    print_tree(simplify(_forest, lambda x: int(x / bucket_size) * bucket_size))
