'''Common functions and data structures used for simplifying trees.'''

from onetree_common import Forest, Tree, Split, Leaf


def collapse_leaf_forest(forest):
    """If forest is nothing but leaves, replace with single leaf."""
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
