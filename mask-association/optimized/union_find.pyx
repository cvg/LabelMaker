import numpy as np


cdef class UnionFind:
    """
    Union Find data structure implemented with union by rank and path compression. 
    """
    cdef public num_elements
    cdef public num_components
    cdef public int[:] parents
    cdef public int[:] sizes

    def __cinit__(self, int num_elements):
        self.num_elements   = num_elements
        self.num_components = num_elements
        self.parents        = -np.ones(num_elements, dtype=np.int32)
        self.sizes          =  np.ones(num_elements, dtype=np.int32)

    def __len__(self):
        return self.num_components

    cpdef int find(self, int i) except *:
        if self.parents[i] == -1:
            return i
        self.parents[i] = self.find(self.parents[i])
        return self.parents[i]

    cpdef void union(self, int i, int j) except *:
        cdef int i_root = self.find(i)
        cdef int j_root = self.find(j)
        if i_root == j_root:
            return
        self.num_components -= 1
        if self.sizes[i_root] < self.sizes[j_root]:
            i_root, j_root = j_root, i_root
        self.parents[j_root] = i_root
        self.sizes[i_root] += self.sizes[j_root]

    cpdef str to_string(self):
        """
        Returns string representation of UnionFind.
        """
        pstr = str([f'{x:02}' for x in self.parents])
        sstr = str([f'{x:02}' for x in self.sizes])
        return f"""
            num_components: {self.num_components}
            parents       : {pstr}
            sizes         : {sstr} 
        """
