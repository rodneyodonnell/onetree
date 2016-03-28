'''Simplify strategy which attempts to maximally reduce runtime at every step.

## Implemented approach

The approach implemented here is:
 1. Find the feature, F, which is most most frequently split on in a random
    path through the forest (assume each branch in a split carries equal traffic).
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
a theoretical ~50x speedup.

In practice, to fully colapse a tree using this algorithm we need to explore
all potential branches in the tree, worst case this may be the product of
the number of split points per variable (i.e., very expensive).

This approach is only useful in a very limited domain of small forests.
'''

import math

from collections import namedtuple
from collections import Counter
from collections import defaultdict

from onetree_common import Forest, Tree, Split, Leaf
from simplify_common import collapse_leaf_forest, merge_buckets

WeightedSplit = namedtuple('WeightedSplit', 'weight split')


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
        'Calculate reducton in expected runtme by splitting on feat.'
        return weights[feat] - math.log(len(cutpoints[feat]), 2)
    best = max((feat for feat in cutpoints), key=saving)

    sorted_cuts = sorted(cutpoints[best])
    return [Split(best, low, high)
            for low, high in zip(sorted_cuts, sorted_cuts[1:])]



def filter_by_split(node, split):
    """Strip out all brances not consistent with 'split'.
    NOTE: Function assumes split() will fit inside exactly one child split, or ValueError thrown.
    """
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


def simplify(forest, bucketize_fn=None, depth=0):
    """Recursively simplify forest into a single tree."""
    weighted_splits = list(get_weighted_splits(forest))

    # Nothing so split on, so we know only a forest of leaves remains.
    if not weighted_splits:
        return collapse_leaf_forest(forest)

    # Turn forest into a tree, where each child is a forest.
    splits = find_best_split(weighted_splits)
    if depth < 6:
        print('\ndepth', depth)
        print('weighted_splits ', len(weighted_splits))
        print('\n'.join(str(s) for s in weighted_splits[:10]))
        print('splits', len(splits))
        print('\n'.join((str(s) for s in splits[:10])))

    children = [simplify(filter_by_split(forest, split), bucketize_fn, depth + 1)
                for split in splits]

    splits, children = merge_buckets(splits, children, bucketize_fn)
    if len(splits) > 1:
        return Tree(splits[0].feature, splits, children)
    if len(splits) == 1:
        return children[0]
