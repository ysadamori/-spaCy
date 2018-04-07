# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
# coding: utf-8
from __future__ import unicode_literals

from cpython.ref cimport Py_INCREF
from cymem.cymem cimport Pool
from collections import OrderedDict, defaultdict, Counter
from thinc.extra.search cimport Beam
import json
from libcpp.vector cimport vector
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy

from ..typedefs cimport hash_t
from ..strings cimport hash_string
from .stateclass cimport StateClass
from ._state cimport StateC, TokenIndexC
from . import nonproj
from .transition_system cimport move_cost_func_t, label_cost_func_t
from ..gold cimport GoldParse, GoldParseC
from ..structs cimport TokenC

# Calculate cost as gold/not gold. We don't use scalar value anyway.
cdef int BINARY_COSTS = 1
cdef weight_t MIN_SCORE = -90000
cdef hash_t SUBTOK_LABEL = hash_string('subtok')

# Sets NON_MONOTONIC, USE_BREAK, USE_SPLIT, MAX_SPLIT
include "compile_time.pxi"

# Break transition inspired by this paper:
# http://www.aclweb.org/anthology/P13-1074
# The most relevant factor is whether we predict Break early, or late:
# do we wait until the root is on the stack, or do we predict when the last
# word of the previous sentence is on the stack?
# The paper applies Break early. This makes life harder, but we find it's
# worth it to give the model flexibility, and Break when stack may be deep.
cdef enum:
    SHIFT
    REDUCE
    LEFT
    RIGHT

    BREAK

    SPLIT

    N_MOVES


MOVE_NAMES = [None] * N_MOVES
MOVE_NAMES[SHIFT] = 'S'
MOVE_NAMES[REDUCE] = 'D'
MOVE_NAMES[LEFT] = 'L'
MOVE_NAMES[RIGHT] = 'R'
MOVE_NAMES[BREAK] = 'B'
MOVE_NAMES[SPLIT] = 'P'


# Helper functions for the arc-eager oracle

cdef weight_t push_cost(StateClass stcls, const GoldParseC* gold, int target) nogil:
    if stcls.c.B(0).i == -1:
        return 0
    cdef weight_t cost = 0
    cdef int i, s_i
    cdef TokenIndexC S_idx
    for i in range(stcls.c.stack_depth()):
        S_idx = stcls.c.S(i)
        s_i = S_idx.j * stcls.c.length + S_idx.i
        if gold.has_dep[target] and gold.heads[target] == s_i:
            cost += 1
        if gold.has_dep[s_i] and gold.heads[s_i] == target and (NON_MONOTONIC or not stcls.has_head(S_idx)):
            cost += 1
        if BINARY_COSTS and cost >= 1:
            return cost
    b0 = stcls.c.B(0).j * stcls.c.length + stcls.c.B(0).i
    if stcls.c.B(0).i >= 0 and gold.has_dep[b0] and Break.is_valid(stcls.c, 0):
        cost += Break.move_cost(stcls, gold) == 0
    # If the token wasn't split before, but gold says it *should* be split,
    # don't push (split instead)
    if Split.is_valid(stcls.c, 0):
        cost += Split.move_cost(stcls, gold) == 0
    return cost


cdef weight_t pop_cost(StateClass stcls, const GoldParseC* gold, int target) nogil:
    cdef weight_t cost = 0
    cdef int i
    cdef TokenIndexC B_idx
    cdef int b_i
    # Take into account fused tokens
    cdef int target_token = target % stcls.c.length
    for i in range(stcls.c.segment_length()):
        B_idx = stcls.c.B(i)
        b_i = B_idx.j * stcls.c.length + B_idx.i
        if gold.has_dep[target]:
            cost += gold.heads[target] == b_i
        if gold.has_dep[b_i]:
            cost += gold.heads[b_i] == target
            if gold.heads[b_i] == b_i or (gold.heads[b_i]%stcls.c.length) < target:
                break
        if BINARY_COSTS and cost >= 1:
            return cost
    return cost


cdef weight_t arc_cost(StateClass stcls, const GoldParseC* gold, int head, int child) nogil:
    cdef TokenIndexC child_idx
    child_idx.i = child % stcls.c.length
    child_idx.j = child // stcls.c.length
    cdef TokenIndexC curr_head = stcls.c.H(child_idx)
    if arc_is_gold(gold, head, child):
        return 0
    elif curr_head.j * stcls.c.length + curr_head.i == gold.heads[child]:
        return 1
    # Head in buffer
    elif gold.heads[child] >= (stcls.c.B(0).i) and stcls.c.B(1).i != 0:
        return 1
    else:
        return 0


cdef bint arc_is_gold(const GoldParseC* gold, int head, int child) nogil:
    if not gold.has_dep[child]:
        return True
    elif gold.heads[child] == head:
        return True
    else:
        return False


cdef bint label_is_gold(const GoldParseC* gold, int head, int child, attr_t label) nogil:
    if not gold.has_dep[child]:
        return True
    elif label == 0:
        return True
    elif gold.labels[child] == label:
        return True
    else:
        return False


cdef bint _is_gold_root(const GoldParseC* gold, int word) nogil:
    return gold.heads[word] == word or not gold.has_dep[word]


cdef class Shift:
    @staticmethod
    cdef bint is_valid(const StateC* st, attr_t label) nogil:
        if st.buffer_length == 0:
            return 0
        elif st.stack_depth() == 0:
            return 1
        elif st.was_shifted(st.B(0)):
            return 0
        elif st.at_break():
            return 0
        else:
            return 1

    @staticmethod
    cdef int transition(StateC* st, attr_t label) nogil:
        b0 = st.B(0).j * st.length + st.B(0).i
        st._shifted[b0] = 1
        st.push()

    @staticmethod
    cdef weight_t cost(StateClass st, const GoldParseC* gold, attr_t label) nogil:
        return Shift.move_cost(st, gold) + Shift.label_cost(st, gold, label)

    @staticmethod
    cdef inline weight_t move_cost(StateClass s, const GoldParseC* gold) nogil:
        return push_cost(s, gold, s.c.B(0).j * s.c.length + s.c.B(0).i)

    @staticmethod
    cdef inline weight_t label_cost(StateClass s, const GoldParseC* gold, attr_t label) nogil:
        return 0


cdef class Split:
    @staticmethod
    cdef bint is_valid(const StateC* st, attr_t label) nogil:
        if not USE_SPLIT:
            return 0
        elif st.buffer_length == 0:
            return 0
        elif st.was_split(st.B(0)):
            return 0
        elif st.B_(0).lex.length == 1:
            return 0
        else:
            return 1

    @staticmethod
    cdef int transition(StateC* st, attr_t label) nogil:
        st.split(0, 1)

    @staticmethod
    cdef weight_t cost(StateClass st, const GoldParseC* gold, attr_t label) nogil:
        return Split.move_cost(st, gold) + Split.label_cost(st, gold, label)

    @staticmethod
    cdef weight_t move_cost(StateClass st, const GoldParseC* gold) nogil:
        if not USE_SPLIT:
            return 9000
        elif st.c.B(0).i < 0:
            return 9000
        elif gold.fused[st.c.B(0).i]:
            return 0
        else:
            return 1

    @staticmethod
    cdef weight_t label_cost(StateClass st, const GoldParseC* gold, attr_t label) nogil:
        if not USE_SPLIT:
            return 9000
        elif gold.fused[st.c.B(0).i]:
            return 0
        else:
            return 1


cdef class Reduce:
    @staticmethod
    cdef bint is_valid(const StateC* st, attr_t label) nogil:
        if st.stack_depth() >= 2:
            return 1
        elif st.at_break() and st.stack_depth() == 1:
            return 1
        else:
            return 0

    @staticmethod
    cdef int transition(StateC* st, attr_t label) nogil:
        if st.has_head(st.S(0)):
            st.pop()
        elif st.stack_depth() == 1 and st.at_break():
            st.pop()
        else:
            st.unshift()

    @staticmethod
    cdef weight_t cost(StateClass s, const GoldParseC* gold, attr_t label) nogil:
        return Reduce.move_cost(s, gold) + Reduce.label_cost(s, gold, label)

    @staticmethod
    cdef inline weight_t move_cost(StateClass st, const GoldParseC* gold) nogil:
        s0 = st.c.S(0).j * st.c.length + st.c.S(0).i
        cost = pop_cost(st, gold, s0)
        cdef TokenIndexC S_i
        cdef int si, i
        if not st.c.has_head(st.c.S(0)):
            # Decrement cost for the arcs we save
            for i in range(1, st.c.stack_depth()):
                S_i = st.c.S(i)
                si = S_i.j * st.c.length + S_i.i
                if gold.heads[s0] == si:
                    cost -= 1
                if gold.heads[si] == s0:
                    cost -= 1
        return cost

    @staticmethod
    cdef inline weight_t label_cost(StateClass s, const GoldParseC* gold, attr_t label) nogil:
        return 0


cdef class LeftArc:
    @staticmethod
    cdef bint is_valid(const StateC* st, attr_t label) nogil:
        if st.buffer_length == 0:
            return 0
        elif st.stack_depth() == 0:
            return 0
        elif st.at_break():
            return 0
        else:
            return 1

    @staticmethod
    cdef int transition(StateC* st, attr_t label) nogil:
        st.add_arc(st.B(0), st.S(0), label)
        st.pop()

    @staticmethod
    cdef weight_t cost(StateClass s, const GoldParseC* gold, attr_t label) nogil:
        cdef weight_t move_cost = LeftArc.move_cost(s, gold)
        cdef weight_t label_cost = LeftArc.label_cost(s, gold, label)
        return move_cost + label_cost

    @staticmethod
    cdef inline weight_t move_cost(StateClass s, const GoldParseC* gold) nogil:
        cdef weight_t cost = 0
        cdef int b0 = s.c.B(0).j * s.c.length + s.c.B(0).i
        cdef int s0 = s.c.S(0).j * s.c.length + s.c.S(0).i
        # Try to repair incorrect split move, using the merge (L-subtok) move.
        if s.c.was_split(s.c.S(0)) and not gold.fused[s0] and s.c.S(0).i == s.c.B(0).i:
            return 0
        if arc_is_gold(gold, b0, s0):
            # Have a negative cost if we 'recover' from the wrong dependency
            return 0 if not s.c.has_head(s.c.S(0)) else -1
        else:
            # Account for deps we might lose between S0 and stack
            if not s.c.has_head(s.c.S(0)):
                for i in range(1, s.c.stack_depth()):
                    si = s.c.S(i).j * s.c.length + s.c.S(i).i
                    cost += gold.heads[si] == s0
                    cost += gold.heads[s0] == si
            return cost + pop_cost(s, gold, s0) + arc_cost(s, gold, b0, s0)

    @staticmethod
    cdef inline weight_t label_cost(StateClass s, const GoldParseC* gold, attr_t label) nogil:
        cdef int b0 = s.c.B(0).j * s.c.length + s.c.B(0).i
        cdef int s0 = s.c.S(0).j * s.c.length + s.c.S(0).i
        # If we're dealing with an incorrect split, ensure label is SUBTOK.
        if s.c.was_split(s.c.S(0)) \
        and not gold.fused[s0] \
        and s.c.S(0).i == s.c.B(0).i:
            return 0 if label == SUBTOK_LABEL else 1
        return arc_is_gold(gold, b0, s0) and not label_is_gold(gold, b0, s0, label)


cdef class RightArc:
    @staticmethod
    cdef bint is_valid(const StateC* st, attr_t label) nogil:
        if st.stack_depth() < 1:
            return 0
        elif st.buffer_length == 0:
            return 0
        elif st.at_break():
            return 0
        # If there's (perhaps partial) parse pre-set, don't allow cycle.
        elif st.has_head(st.S(0)) and st.H(st.S(0)).i == st.B(0).i:
            return 0
        else:
            return 1

    @staticmethod
    cdef int transition(StateC* st, attr_t label) nogil:
        st.add_arc(st.S(0), st.B(0), label)
        st.push()

    @staticmethod
    cdef inline weight_t cost(StateClass s, const GoldParseC* gold, attr_t label) nogil:
        # TODO: Handle oracle for incorrect splits
        return RightArc.move_cost(s, gold) + RightArc.label_cost(s, gold, label)

    @staticmethod
    cdef inline weight_t move_cost(StateClass s, const GoldParseC* gold) nogil:
        cdef int b0 = s.c.B(0).j * s.c.length + s.c.B(0).i
        cdef int s0 = s.c.S(0).j * s.c.length + s.c.S(0).i
        # If the token wasn't split before, but gold says it *should* be split,
        # don't right-arc (split instead)
        if USE_SPLIT and not s.c.was_split(s.c.B(0)) and gold.fused[b0]:
            return gold.fused[b0]
        elif arc_is_gold(gold, s0, b0):
            return 0
        elif s.c.was_shifted(s.c.B(0)):
            return push_cost(s, gold, b0)
        else:
            return push_cost(s, gold, b0) + arc_cost(s, gold, s0, b0)

    @staticmethod
    cdef weight_t label_cost(StateClass s, const GoldParseC* gold, attr_t label) nogil:
        cdef int b0 = s.c.B(0).j * s.c.length + s.c.B(0).i
        cdef int s0 = s.c.S(0).j * s.c.length + s.c.S(0).i
        return arc_is_gold(gold, s0, b0) and not label_is_gold(gold, s0, b0, label)


cdef class Break:
    @staticmethod
    cdef bint is_valid(const StateC* st, attr_t label) nogil:
        # It would seem good to have a stack_depth==1 constraint here.
        # That would make the other validities much less complicated.
        # However, we need to know about upcoming sentence break to respect
        # preset SBD anyway --- so we may as well give the parser the flexibility.
        cdef int i
        if not USE_BREAK:
            return 0
        elif st.buffer_length == 0:
            return 0
        elif st.stack_depth() < 1:
            return 0
        elif st._sent[st.B_(0).l_edge].sent_start == -1:
            return 0
        elif st.B(0).j != 0:
            return 0
        else:
            return 1

    @staticmethod
    cdef int transition(StateC* st, attr_t label) nogil:
        st.set_break(0)
        st.pop()

    @staticmethod
    cdef weight_t cost(StateClass s, const GoldParseC* gold, attr_t label) nogil:
        return Break.move_cost(s, gold) + Break.label_cost(s, gold, label)

    @staticmethod
    cdef inline weight_t move_cost(StateClass s, const GoldParseC* gold) nogil:
        cdef weight_t cost = 0
        cdef int i
        cdef TokenIndexC S_i, B_i
        cdef int s0 = s.c.S(0).j * s.c.length + s.c.S(0).i
        cdef int b0 = s.c.B(0).j * s.c.length + s.c.B(0).i
        cdef int si, bi
        for i in range(s.c.stack_depth()):
            S_i = s.c.S(i)
            si = S_i.j * s.c.length + S_i.i
            for j in range(s.c.buffer_length):
                B_i = s.c.B(j)
                bi = B_i.j * s.c.length + B_i.i
                if not gold.has_dep[si]:
                    cost += gold.heads[si] == bi
                if gold.has_dep[bi]:
                    cost += gold.heads[bi] == si
                if cost != 0:
                    return cost
        # Check for sentence boundary --- if it's here, we can't have any deps
        # between stack and buffer, so rest of action is irrelevant.
        s0_root = _get_root(s0, gold)
        b0_root = _get_root(b0, gold)
        if s0_root != b0_root or s0_root == -1 or b0_root == -1:
            return cost
        else:
            return cost + 1

    @staticmethod
    cdef inline weight_t label_cost(StateClass s, const GoldParseC* gold, attr_t label) nogil:
        return 0

cdef int _get_root(int word, const GoldParseC* gold) nogil:
    while gold.heads[word] != word and gold.has_dep[word] and word >= 0:
        word = gold.heads[word]
    if not gold.has_dep[word]:
        return -1
    else:
        return word


cdef void* _init_state(Pool mem, int length, void* tokens) except NULL:
    st = new StateC(<const TokenC*>tokens, length)
    for i in range(st.length):
        if st._sent[i].dep == 0:
            st._sent[i].l_edge = i
            st._sent[i].r_edge = i
            st._sent[i].head = 0
            st._sent[i].dep = 0
            st._sent[i].l_kids = 0
            st._sent[i].r_kids = 0
    return <void*>st


cdef class ArcEager(TransitionSystem):
    def __init__(self, *args, **kwargs):
        TransitionSystem.__init__(self, *args, **kwargs)
        self.init_beam_state = _init_state
        if USE_SPLIT:
            for i in range(1, MAX_SPLIT):
                self.add_action(SPLIT, str(i))

    @classmethod
    def get_actions(cls, **kwargs):
        min_freq = kwargs.get('min_freq', None)
        actions = defaultdict(lambda: Counter())
        actions[SHIFT][''] = 1
        actions[REDUCE][''] = 1
        for label in kwargs.get('left_labels', []):
            actions[LEFT][label] = 1
            actions[SHIFT][label] = 1
        for label in kwargs.get('right_labels', []):
            actions[RIGHT][label] = 1
            actions[REDUCE][label] = 1
        for raw_text, sents in kwargs.get('gold_parses', []):
            for (ids, words, tags, heads, labels, iob), ctnts in sents:
                heads, labels = nonproj.projectivize(heads, labels)
                for child, head, label in zip(ids, heads, labels):
                    if label.upper() == 'ROOT' :
                        label = 'ROOT'
                    if head == child:
                        actions[BREAK][label] += 1
                    elif head < child:
                        actions[RIGHT][label] += 1
                        actions[REDUCE][''] += 1
                    elif head > child:
                        actions[LEFT][label] += 1
                        actions[SHIFT][''] += 1
        if min_freq is not None:
            for action, label_freqs in actions.items():
                for label, freq in list(label_freqs.items()):
                    if freq < min_freq:
                        label_freqs.pop(label)
        # Ensure these actions are present
        actions[BREAK].setdefault('ROOT', 0)
        actions[RIGHT].setdefault('subtok', 0)
        actions[LEFT].setdefault('subtok', 0)
        # Used for backoff
        actions[RIGHT].setdefault('dep', 0)
        actions[LEFT].setdefault('dep', 0)
        # TODO: Split?
        return actions

    property max_split:
        def __get__(self):
            if not USE_SPLIT:
                return 0
            else:
                return MAX_SPLIT

    property action_types:
        def __get__(self):
            if USE_SPLIT:
                return (SHIFT, REDUCE, LEFT, RIGHT, BREAK, SPLIT)
            else:
                return (SHIFT, REDUCE, LEFT, RIGHT, BREAK)

    def get_cost(self, StateClass state, GoldParse gold, action):
        cdef Transition t = self.lookup_transition(action)
        if not t.is_valid(state.c, t.label):
            return 9000
        else:
            return t.get_cost(state, &gold.c, t.label)

    def transition(self, StateClass state, action):
        cdef Transition t = self.lookup_transition(action)
        t.do(state.c, t.label)
        return state
 
    def is_gold_parse(self, StateClass state, GoldParse gold):
        predicted = set()
        truth = set()
        cdef TokenIndexC idx
        for i in range(gold.length):
            gold_i = gold._alignment.index_to_yours(i)
            idx.i = i
            idx.j = 0
            if gold_i is None:
                continue
            if state.c.safe_get(idx).dep:
                predicted.add((i, state.c.H(idx).i,
                              self.strings[state.c.safe_get(idx).dep]))
            else:
                predicted.add((i, state.c.H(idx).i, 'ROOT'))
            id_, word, tag, head, dep, ner = gold.orig_annot[gold_i]
            truth.add((id_, head, dep))
        return truth == predicted

    def has_gold(self, GoldParse gold, start=0, end=None):
        end = end or len(gold.heads)
        if all([tag is None for tag in gold.heads[start:end]]):
            return False
        else:
            return True

    def preprocess_gold(self, GoldParse gold):
        if not self.has_gold(gold):
            return None
        subtok_label = self.strings['subtok']
        if USE_SPLIT:
            gold.resize_arrays(MAX_SPLIT * len(gold))
            for i in range(len(gold)):
                if isinstance(gold.heads[i], list) and len(gold.heads[i]) <= MAX_SPLIT:
                    gold.c.fused[i] = len(gold.heads[i])-1
                else:
                    gold.c.fused[i] = 0

        heads, labels = self._fix_heads_labels_format(gold)
        for child_i, (head_group, dep_group) in enumerate(zip(heads, labels)):
            for child_j, (head_addr, dep) in enumerate(zip(head_group, dep_group)):
                head_i, head_j = head_addr
                child_index = child_j * len(gold) + child_i
                # Missing values
                if head_i is None or dep is None:
                    gold.c.heads[child_index] = child_index
                    gold.c.has_dep[child_index] = False
                    continue
                head_index = head_j * len(gold) + head_i
                if (head_i, head_j) > (child_i, child_j):
                    action = LEFT
                elif (head_i, head_j) < (child_i, child_j):
                    action = RIGHT
                else:
                    action = BREAK
                if dep not in self.labels[action]:
                    dep = self._backoff_label(dep, action)
                gold.c.has_dep[child_index] = True
                if dep.upper() == 'ROOT':
                    dep = 'ROOT'
                gold.c.heads[child_index] = head_index
                gold.c.labels[child_index] = self.strings.add(dep)
        return gold

    def _fix_heads_labels_format(self, gold):
        heads = []
        labels = []
        for child_i, (head_group, dep_group) in enumerate(zip(gold.heads, gold.labels)):
            if not USE_SPLIT and (isinstance(head_group, list) or isinstance(head_group, tuple)):
                # Set as missing values if we don't handle token splitting
                head_group = [(None, 0)]
                dep_group = [None]
            if isinstance(head_group, list) and len(head_group) > MAX_SPLIT:
                head_group = [(None, 0)]
                dep_group = [None]
            if not isinstance(head_group, list):
                # Map the simple format into the elaborate one we need for
                # the fused tokens.
                head_group = [head_group]
                dep_group = [dep_group]
            heads.append([])
            labels.append([])
            for child_j, (head_addr, dep) in enumerate(zip(head_group, dep_group)):
                if not isinstance(head_addr, tuple):
                    head_addr = (head_addr, 0)
                head_i, head_j = head_addr
                if not USE_SPLIT:
                    head_j = 0
                    child_j = 0
                elif head_j >= MAX_SPLIT:
                    head_i = None
                    dep = None
                heads[-1].append((head_i, head_j))
                labels[-1].append(dep)
        return heads, labels

    def _backoff_label(self, dep, action):
        if action == BREAK:
            return 'ROOT'
        elif not nonproj.is_decorated(dep):
            return 'dep'
        backoff = nonproj.decompose(dep)[0]
        return backoff if backoff in self.labels[action] else 'dep'

    def get_beam_parses(self, Beam beam):
        parses = []
        probs = beam.probs
        for i in range(beam.size):
            state = <StateC*>beam.at(i)
            if state.is_final():
                self.finalize_state(state)
                prob = probs[i]
                parse = []
                for j in range(state.length):
                    head = state.H(TokenIndexC(i=j, j=0))
                    label = self.strings[state._sent[j].dep]
                    parse.append((head, j, label))
                parses.append((prob, parse))
        return parses

    cdef Transition lookup_transition(self, object name_or_id) except *:
        if isinstance(name_or_id, int):
            return self.c[name_or_id]
        name = name_or_id
        if '-' in name:
            move_str, label_str = name.split('-', 1)
            label = self.strings[label_str]
        else:
            move_str = name
            label = 0
        move = MOVE_NAMES.index(move_str)
        for i in range(self.n_moves):
            if self.c[i].move == move and self.c[i].label == label:
                return self.c[i]
        return Transition(clas=0, move=MISSING, label=0, score=0.)

    def move_name(self, int move, attr_t label):
        label_str = self.strings[label]
        if label_str:
            return MOVE_NAMES[move] + '-' + label_str
        else:
            return MOVE_NAMES[move]

    def class_name(self, int i):
        return self.move_name(self.c[i].move, self.c[i].label)

    cdef Transition init_transition(self, int clas, int move, attr_t label) except *:
        # TODO: Apparent Cython bug here when we try to use the Transition()
        # constructor with the function pointers
        cdef Transition t
        t.score = 0
        t.clas = clas
        t.move = move
        t.label = label
        if move == SHIFT:
            t.is_valid = Shift.is_valid
            t.do = Shift.transition
            t.get_cost = Shift.cost
        elif move == REDUCE:
            t.is_valid = Reduce.is_valid
            t.do = Reduce.transition
            t.get_cost = Reduce.cost
        elif move == LEFT:
            t.is_valid = LeftArc.is_valid
            t.do = LeftArc.transition
            t.get_cost = LeftArc.cost
        elif move == RIGHT:
            t.is_valid = RightArc.is_valid
            t.do = RightArc.transition
            t.get_cost = RightArc.cost
        elif move == BREAK:
            t.is_valid = Break.is_valid
            t.do = Break.transition
            t.get_cost = Break.cost
        elif move == SPLIT:
            t.is_valid = Split.is_valid
            t.do = Split.transition
            t.get_cost = Split.cost
        else:
            raise Exception(move)
        return t

    cdef int initialize_state(self, StateC* st) nogil:
        for i in range(st.length):
            if st._sent[i].dep == 0:
                st._sent[i].l_edge = i
                st._sent[i].r_edge = i
                st._sent[i].head = 0
                st._sent[i].dep = 0
                st._sent[i].l_kids = 0
                st._sent[i].r_kids = 0

    def _py_finalize_state(self, StateClass state):
        self.finalize_state(state.c)

    cdef int finalize_state(self, StateC* st) nogil:
        cdef int i, j
        for i in range(st.length):
            if st._sent[i].head == 0:
                st._sent[i].dep = self.root_label
        # Resolve split tokens back into the sentence
        # First we gather a list of the indices into the state._sent array,
        # in the order they should occur.
        cdef vector[int] indices
        cdef vector[int] old2new
        # Make an index map so we can fix heads
        old2new.resize(st.length * MAX_SPLIT)
        for i in range(st.length):
            old2new[i] = indices.size()
            indices.push_back(i)
            for j in range(1, st._was_split[i]+1):
                old2new[j*st.length + i] = indices.size()
                indices.push_back(j*st.length+i)
        # Now make a copy of the array, so we can set the correct order into
        # st._sent
        old = <TokenC*>calloc(st.length * MAX_SPLIT, sizeof(TokenC))
        memcpy(old, st._sent, st.length * MAX_SPLIT * sizeof(TokenC))
        # Now set the array.
        cdef int prev_idx, old_head
        for i in range(indices.size()):
            prev_idx = indices[i]
            st._sent[i] = old[prev_idx]
            if prev_idx >= st.length:
                st._sent[i].lex = st._empty_token.lex
            old_head = prev_idx + old[prev_idx].head
            st._sent[i].head = old2new[old_head] - i
        st.length = indices.size()
        free(old)

    def finalize_doc(self, doc):
        doc.is_parsed = True
        for sent in doc.sents:
            for word in sent:
                if word.head.i == word.i and word.dep_ == 'ROOT':
                    break
            else:
                print("Rootless sentence!")
                print(sent)
                for w in sent:
                    print(w.i, w.text, w.head.text, w.head.i, w.dep_)
                raise ValueError



    cdef int set_valid(self, int* output, const StateC* st) nogil:
        cdef bint[N_MOVES] is_valid
        is_valid[SHIFT] = Shift.is_valid(st, 0)
        is_valid[REDUCE] = Reduce.is_valid(st, 0)
        is_valid[LEFT] = LeftArc.is_valid(st, 0)
        is_valid[RIGHT] = RightArc.is_valid(st, 0)
        is_valid[BREAK] = Break.is_valid(st, 0)
        is_valid[SPLIT] = Split.is_valid(st, 0)
        cdef int i
        for i in range(self.n_moves):
            output[i] = is_valid[self.c[i].move]

    cdef int set_costs(self, int* is_valid, weight_t* costs,
                       StateClass stcls, GoldParse gold) except -1:
        cdef int i, move
        cdef attr_t label
        cdef label_cost_func_t[N_MOVES] label_cost_funcs
        cdef move_cost_func_t[N_MOVES] move_cost_funcs
        cdef weight_t[N_MOVES] move_costs
        for i in range(N_MOVES):
            move_costs[i] = 9000
        move_cost_funcs[SHIFT] = Shift.move_cost
        move_cost_funcs[REDUCE] = Reduce.move_cost
        move_cost_funcs[LEFT] = LeftArc.move_cost
        move_cost_funcs[RIGHT] = RightArc.move_cost
        move_cost_funcs[BREAK] = Break.move_cost
        move_cost_funcs[SPLIT] = Split.move_cost

        label_cost_funcs[SHIFT] = Shift.label_cost
        label_cost_funcs[REDUCE] = Reduce.label_cost
        label_cost_funcs[LEFT] = LeftArc.label_cost
        label_cost_funcs[RIGHT] = RightArc.label_cost
        label_cost_funcs[BREAK] = Break.label_cost
        label_cost_funcs[SPLIT] = Split.label_cost

        cdef attr_t* labels = gold.c.labels
        cdef int* heads = gold.c.heads

        n_gold = 0
        for i in range(self.n_moves):
            if self.c[i].is_valid(stcls.c, self.c[i].label):
                is_valid[i] = True
                move = self.c[i].move
                label = self.c[i].label
                if move_costs[move] == 9000:
                    move_costs[move] = move_cost_funcs[move](stcls, &gold.c)
                costs[i] = move_costs[move] + label_cost_funcs[move](stcls, &gold.c, label)
                n_gold += costs[i] <= 0
            else:
                is_valid[i] = False
                costs[i] = 9000
        if n_gold < 1:
            print("Heads for length", len(gold))
            print([(i, gold.c.heads[i]) for i in range(len(gold))])
            print(gold.words)
            print(gold.labels)
            print(list(gold.heads))
            print(list(enumerate(gold.labels)))
            # Check label set --- leading cause
            label_set = set([self.strings[self.c[i].label] for i in range(self.n_moves)])
            for label_str in gold.labels:
                if isinstance(label_str, list):
                    continue
                if label_str is not None and label_str not in label_set:
                    raise ValueError("Cannot get gold parser action: unknown label: %s" % label_str)
            # Check projectivity --- other leading cause
            if nonproj.is_nonproj_tree(gold._alignment.flatten(gold.heads)):
                raise ValueError(
                    "Could not find a gold-standard action to supervise the "
                    "dependency parser. Likely cause: the tree is "
                    "non-projective (i.e. it has crossing arcs -- see "
                    "spacy/syntax/nonproj.pyx for definitions). The ArcEager "
                    "transition system only supports projective trees. To "
                    "learn non-projective representations, transform the data "
                    "before training and after parsing. Either pass "
                    "make_projective=True to the GoldParse class, or use "
                    "spacy.syntax.nonproj.preprocess_training_data.")
            else:
                print(gold.orig_annot)
                print(gold.words)
                print(gold.heads)
                print(gold.labels)
                print(gold.sent_starts)
                print(stcls.c.stack_depth(), stcls.stack, stcls.queue)
                print(stcls.c.B(0).i, stcls.c.B(0).j)
                print(stcls.c.S(0).i, stcls.c.S(0).j)
                print(stcls.history)
                raise ValueError(
                    "Could not find a gold-standard action to supervise the"
                    "dependency parser. The GoldParse was projective. The "
                    "transition system has %d actions. State at failure: %s"
                    % (self.n_moves, stcls.print_state(map(repr, gold.words))))
        assert n_gold >= 1

    def get_beam_annot(self, Beam beam):
        length = (<StateC*>beam.at(0)).length
        heads = [{} for _ in range(length)]
        deps = [{} for _ in range(length)]
        probs = beam.probs
        for i in range(beam.size):
            state = <StateC*>beam.at(i)
            self.finalize_state(state)
            if state.is_final():
                prob = probs[i]
                for j in range(state.length):
                    head = j + state._sent[j].head
                    dep = state._sent[j].dep
                    heads[j].setdefault(head, 0.0)
                    heads[j][head] += prob
                    deps[j].setdefault(dep, 0.0)
                    deps[j][dep] += prob
        return heads, deps


