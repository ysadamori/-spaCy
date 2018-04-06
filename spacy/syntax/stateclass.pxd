from libc.string cimport memcpy, memset

from cymem.cymem cimport Pool
cimport cython

from ..structs cimport TokenC, Entity
from ..typedefs cimport attr_t

from ..vocab cimport EMPTY_LEXEME
from ._state cimport StateC, TokenIndexC


@cython.final
cdef class StateClass:
    cdef Pool mem
    cdef StateC* c
    cdef int _borrowed

    @staticmethod
    cdef inline StateClass init(const TokenC* sent, int length):
        cdef StateClass self = StateClass()
        self.c = new StateC(sent, length)
        return self
    
    @staticmethod
    cdef inline StateClass borrow(StateC* ptr):
        cdef StateClass self = StateClass()
        del self.c
        self.c = ptr
        self._borrowed = 1
        return self


    @staticmethod
    cdef inline StateClass init_offset(const TokenC* sent, int length, int
                                       offset):
        cdef StateClass self = StateClass()
        self.c = new StateC(sent, length)
        self.c.offset = offset
        return self
