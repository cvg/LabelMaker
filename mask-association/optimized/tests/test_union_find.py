import pytest
import numpy as np
import union_find


@pytest.fixture
def dsu():
    return union_find.UnionFind(10)


def test_empty(dsu):
    assert len(dsu) == 10
    assert dsu.num_components == 10
    assert dsu.find(0) == 0
    assert dsu.find(1) == 1


def test_one(dsu):
    dsu.union(0, 1)
    assert dsu.find(0) == dsu.find(1)
    assert dsu.find(2) != dsu.find(0)
    assert dsu.num_components == 9


def test_many(dsu):
    dsu.union(0, 1)
    print(dsu.to_string())
    dsu.union(2, 3)
    dsu.union(3, 4)
    dsu.union(5, 6)
    print(dsu.to_string())
    assert dsu.num_components == 6
    assert dsu.find(0) == dsu.find(1)
    assert dsu.find(2) == dsu.find(4)
    assert dsu.find(0) != dsu.find(2)
    assert dsu.find(0) != dsu.find(5)
    assert dsu.find(2) != dsu.find(6)
    
    dsu.union(4, 5)
    assert dsu.num_components == 5
    assert dsu.find(0) != dsu.find(2)
    assert dsu.find(2) == dsu.find(6)
    print(dsu.to_string())