"""Script to convert a decision forest into a single tree.

## Implemented approaches

 - greedy_by_runtime:
   Greedily remove variable which saves us the largest expected runtime per
   item.  This hits combinational explosion very quickly for large forests.

## Alternative approach 2 - Split on common vars.

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

from onetree_common import print_tree

import simplify_greedy_runtime


def main():
    '''main(), just build & test for now.'''
    from onetree_common_test import make_small_test_forest
    _forest = make_small_test_forest()

    print('\n## Full Tree')
    print_tree(_forest)

    print('\n## Simplified')
    print_tree(simplify_greedy_runtime.simplify(_forest))

    print('\n## Simplified with buckets')
    bucket_size = 0.1
    print_tree(simplify_greedy_runtime.simplify(_forest, lambda x: int(x / bucket_size) * bucket_size))

if __name__ == '__main__':
    main()
