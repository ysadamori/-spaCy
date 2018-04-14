# coding: utf-8
# cython: infer_types=True
from __future__ import unicode_literals

import numpy
cimport numpy as np
from murmurhash.mrmr cimport hash64
from libc.stdint cimport uint64_t

from ..tokens.doc cimport Doc


cdef class StateClass:
    def __init__(self, Doc doc=None, int offset=0, int max_split=0):
        cdef Pool mem = Pool()
        self.mem = mem
        self._borrowed = 0
        if doc is not None:
            self.c = new StateC(doc.c, doc.length)
            self.c.offset = offset
            self.c.max_split = max_split

    def __dealloc__(self):
        if self._borrowed != 1:
            del self.c

    def __len__(self):
        return self.c.length

    def get_B(self, int i):
        return self.c.B(i).j * self.c.length + self.c.B(i).i
    
    def get_S(self, int i):
        return self.c.S(i).j * self.c.length + self.c.S(i).i

    def can_push(self):
        return self.c.can_push()

    def can_pop(self):
        return self.c.can_pop()

    def can_break(self):
        return self.c.can_break()

    def can_arc(self):
        return self.c.can_arc()

    def push_stack(self):
        self.c.push()

    def pop_stack(self):
        self.c.pop()

    def unshift(self):
        self.c.unshift()

    def set_break(self, int i):
        self.c.set_break(i)

    def split_token(self, int i, int n):
        self.c.split(i, n)

    def get_doc(self, vocab):
        cdef Doc doc = Doc(vocab)
        doc.vocab = vocab
        doc.c = self.c._sent
        doc.length = self.c.length
        return doc

    @property
    def buffer_length(self):
        return self.c.buffer_length

    @property
    def stack(self):
        return [self.get_S(i) for i in range(self.c._s_i)]

    @property
    def queue(self):
        return [self.get_B(i) for i in range(self.c.buffer_length)]

    @property
    def token_vector_lenth(self):
        return self.doc.tensor.shape[1]
    
    @property
    def extra_features(self):
        cdef uint64_t[10] values
        values[0] = self.c.S_(0).dep
        values[1] = self.c.S_(1).dep
        values[2] = self.c.S_(2).dep
        values[3] = self.c.L_(self.c.S(0), 1).dep
        values[4] = self.c.L_(self.c.S(0), 2).dep
        values[5] = self.c.R_(self.c.S(0), 1).dep
        values[6] = self.c.R_(self.c.S(0), 2).dep
        values[7] = self.c.L_(self.c.B(0), 1).dep
        values[8] = self.c.L_(self.c.B(0), 2).dep
        cdef int[3] status
        status[0] = self.c.was_shifted(self.c.B(0))
        status[1] = self.c.stack_depth()
        status[2] = self.c.segment_length()
        values[9] = hash64(status, sizeof(status), 0)
        cdef np.ndarray features = numpy.zeros((10,), dtype='uint64')
        features[0] = hash64(&values[0], sizeof(values[0]), 0)
        features[1] = hash64(&values[1], sizeof(values[1]), 1)
        features[2] = hash64(&values[2], sizeof(values[2]), 2)
        features[3] = hash64(&values[3], sizeof(values[3]), 3)
        features[4] = hash64(&values[4], sizeof(values[4]), 4)
        features[5] = hash64(&values[5], sizeof(values[5]), 5)
        features[6] = hash64(&values[6], sizeof(values[6]), 6)
        features[7] = hash64(&values[7], sizeof(values[7]), 7)
        features[8] = hash64(&values[8], sizeof(values[8]), 8)
        features[9] = hash64(&values[9], sizeof(values[9]), 9)
        return features


    @property
    def history(self):
        hist = numpy.ndarray((8,), dtype='i')
        for i in range(8):
            hist[i] = self.c.get_hist(i+1)
        return hist

    def is_final(self):
        return self.c.is_final()

    def copy(self):
        cdef StateClass new_state = StateClass.init(self.c._sent, self.c.length)
        new_state.c.clone(self.c)
        return new_state

    def print_state(self, words):
        words = list(words) + ['_']
        top = words[self.c.S(0).i] + '_%d' % self.c.S_(0).head
        second = words[self.c.S(1).i] + '_%d' % self.c.S_(1).head
        third = words[self.c.S(2).i] + '_%d' % self.c.S_(2).head
        n0 = words[self.c.B(0).i]
        n1 = words[self.c.B(1).i]
        return ' '.join((third, second, top, '|', n0, n1))
