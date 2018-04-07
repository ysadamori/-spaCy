import pytest

from ...tokens.doc import Doc
from ...vocab import Vocab
from ...syntax.stateclass import StateClass
from ...syntax.arc_eager import ArcEager


def get_doc(words, vocab=None):
    if vocab is None:
        vocab = Vocab()
    return Doc(vocab, words=list(words))

def test_push():
    '''state.push_stack() should take the first word in the queue (aka buffer)
    and put it on the stack, popping that word from the queue.'''
    doc = get_doc('abcd')
    state = StateClass(doc)
    assert state.get_B(0) == 0
    state.push_stack()
    assert state.get_B(0) == 1

def test_pop():
    '''state.pop_stack() should remove the top word from the stack.'''
    doc = get_doc('abcd')
    state = StateClass(doc)
    assert state.get_B(0) == 0
    state.push_stack()
    state.push_stack()
    assert state.get_S(0) == 1
    assert state.get_S(1) == 0
    state.pop_stack()
    assert state.get_S(0) == 0


def test_split():
    '''state.split_token should take the ith word of the buffer, and split it
    into n+1 pieces. n is 0-indexed, i.e. split(i, 0) is a noop, and split(i, 1)
    creates 1 new token.'''
    doc = get_doc('abcd')
    state = StateClass(doc, max_split=3)
    assert state.queue == [0, 1, 2, 3]
    state.split_token(1, 2)
    assert state.queue == [0, 1, 1*4+1, 2*4+1, 2, 3]

def test_finalize_state():
    doc = get_doc('ab')
    M = ArcEager(doc.vocab.strings)
    M.add_action(0, 0)
    M.add_action(1, 0)
    M.add_action(2, 'dep')
    M.add_action(3, 'dep')
    M.add_action(4, 'ROOT')
    M.add_action(5, '1')
    state = StateClass(doc, max_split=M.max_split)
    M.transition(state, 'P-1')
    M.transition(state, 'S')
    M.transition(state, 'R-dep')
    M.transition(state, 'D')
    M.transition(state, 'L-dep')
    M.transition(state, 'S')
    M.transition(state, 'B-ROOT')
    assert state.is_final()
    M._py_finalize_state(state)
    output = state.get_doc(doc.vocab)
    assert len(output) == 3
    assert output[0].head.i == 2
    assert output[1].head.i == 0
    assert output[2].head.i == 2
    assert output[0].dep_ == 'dep'
    assert output[1].dep_ == 'dep'
    assert output[2].dep_ == 'ROOT'
