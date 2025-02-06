import pytest
import numpy as np
import bit

@pytest.fixture
def tree():
    return bit.BIT(10000)


def test_empty(tree):
    assert tree.lower_bound(0) == -1
    assert tree.lower_bound(1) == tree.num_elements
    assert tree.range_sum(0) == 0
    assert tree.range_sum(1) == 0


def test_single(tree):
    tree.update(0, 1)
    assert tree.lower_bound(1) == 0
    assert tree.range_sum(0) == 1
    assert tree.range_sum(1) == 1
    assert tree.lower_bound(1) == 0
    assert tree.lower_bound(2) == tree.num_elements


def test_many(tree):
    tree.update(1, 1)
    tree.update(2, 2)
    assert tree.range_sum(0) == 0
    assert tree.range_sum(1000) == 3
    assert tree.range_sum(1) == 1
    assert tree.range_sum(2) == 3
    assert tree.lower_bound(1) == 1
    assert tree.lower_bound(3) == 2
    assert tree.lower_bound(2) == 2
    assert tree.lower_bound(4) == tree.num_elements
    assert tree.lower_bound(5) == tree.num_elements


def test_many_large_indices(tree):
    tree.update(0, 1)
    tree.update(1000, 1)
    assert tree.range_sum(999) == 1
    assert tree.range_sum(1000) == 2
    assert tree.lower_bound(0) == -1
    assert tree.lower_bound(5) == tree.num_elements

    tree.update(1000, 1)
    tree.update(1000, 2)
    assert tree.range_sum(1000) == 5

    tree.update(2, 2)
    assert tree.range_sum(1000) == 7
    assert tree.range_sum(1001) == 7
    assert tree.range_sum(999) == 3
    assert tree.lower_bound(0) == -1
    assert tree.lower_bound(1) == 0
    assert tree.lower_bound(3) == 2
    assert tree.lower_bound(7) == 1000
    assert tree.lower_bound(8) == tree.num_elements