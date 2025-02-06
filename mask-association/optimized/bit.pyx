import numpy as np


cdef next_pow2(int n):
    """ 
    Returns the next power of 2 of n and log n
    """
    cdef int logn = int(np.log2(n - 1)) + 1
    cdef int pow2 = 1 << logn
    return pow2, logn


cdef class BIT:
    """
    Zero-indexed Binary Indexed Tree that supports the lower_bound operation in O(log n)
    """
    cdef public num_elements
    cdef int N
    cdef int LOGN
    cdef double[:] elements

    def __cinit__(self, int num_elements):
        self.num_elements = num_elements
        self.N, self.LOGN = next_pow2(num_elements)
        self.elements     = np.zeros(self.N + 1, dtype=np.double)

    cpdef void update(self, int i, int v) except *:
        while i <= self.N:
            self.elements[i] += v
            i = i | (i + 1)

    cpdef double range_sum(self, int i) except *:
        cdef double total = 0
        while i >= 0:
            total += self.elements[i]
            i = (i & (i + 1)) - 1
        return total

    cpdef int lower_bound(self, int v) except *:
        if v == 0:
            return -1
        cdef double total = 0
        cdef int index = 0
        cdef int current_index = 0
        cdef int i = self.LOGN
        while i >= 0:
            current_index = index + (1 << i) - 1
            if current_index + 1 < self.N and total + self.elements[current_index] < v:
                total += self.elements[current_index]
                index += (1 << i)
            i -= 1
        return index if index < self.num_elements else self.num_elements

    cpdef str to_string(self):
        """
        Returns string representation of BIT in O(nlogn) 
        """
        return str([self.range_sum(i) for i in range(self.num_elements)])