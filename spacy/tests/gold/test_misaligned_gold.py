'''Test logic for mapping annotations when the gold parse and the doc don't
align in tokenization.'''
from __future__ import unicode_literals
import pytest
from collections import Counter
from ...tokens import Doc
from ...gold import GoldParse
from ...vocab import Vocab


def test_over_segmented():
    doc = Doc(Vocab(),  words=['a', 'b', 'c'])
    gold = GoldParse(doc, words=['ab', 'c'], heads=[1,1])
    assert gold._alignment._y2t == [(0, 0), (0, 1), 1]
    assert gold.heads == [1, 2, 2]
    assert gold.labels == ['subtok', None, None]
                       
                       
def test_under_segmented():
    doc = Doc(Vocab(),  words=['ab', 'c'])
    gold = GoldParse(doc, words=['a', 'b', 'c'], heads=[2,2,2])
    assert gold.heads == [[1,1], 1]
    assert gold.labels == [[None, None], None]
                       
def test_over_segmented_heads():
    doc = Doc(Vocab(),  words=['a', 'b', 'c', 'd', 'e'])
    gold = GoldParse(doc, words=['a', 'bc', 'd', 'e'], heads=[2,2,2,2])
    assert gold._alignment._y2t == [0, (1, 0), (1, 1), 2, 3]
    assert gold._alignment._t2y == [0, [1, 2], 3, 4]
    assert gold.labels == [None, 'subtok', None, None, None]
    assert gold.heads == [3, 2, 3, 3, 3]
 
def test_under_segmented_attach_inside_fused():
    '''Test arcs point ing into the fused token,
    e.g. "its good"    
    '''                
    doc = Doc(Vocab(), words=['ab', 'c'])
    gold = GoldParse(doc, words=['a', 'b', 'c'], heads=[1,1,1])
    assert gold.heads == [[(0, 1), (0, 1)], (0, 1)]
    assert gold.labels == [[None, None], None]
                    

def test_oversegment_example():
    doc = Doc(Vocab(), words=['hunger', '.', '*', '*', 'Edit', ':'])
    gold_words = ['hunger', '.', '**', 'Edit', ':']
    heads = [0, 0, 3, 3, 3]
    deps = ['ROOT', 'punct', 'punct', 'ROOT', 'punct']
    gold = GoldParse(doc, words=gold_words, heads=heads, deps=deps)
    assert gold.words == ['hunger', '.', ('**', 0), ('**', 1), 'Edit', ':']
    assert gold.heads == [0, 0, 3, 4, 4, 4]

    words = ['it', 'runs', 'you', 'about', '4', 'bucks', 'and', 'it', 'deals', 'crushing', 'blows', 'to', 'hunger', '.', '*', '*', 'Edit', ':', 'Living', 'on', 'campus', 'at', 'Clarkson', 'University', ',', 'I', 'have', 'had', 'food', 'delivered', 'before', '.', 'This', 'was', 'back', 'between', "'", '05', 'and', "'", '09', 'and', 'I', 'do', "n't", 'remember', 'how', 'many', 'times', 'we', "'ve", 'had', 'it', 'delivered', '.', 'Perhaps', 'they', 'do', "n't", 'deliver', 'anymore', ',', 'but', 'the', 'deliciousness', 'of', 'a', 'mezza', 'luna', 'certainly', 'warrants', 'a', 'pickup', '.']
    gold_words = ['it', 'runs', 'you', 'about', '4', 'bucks', 'and', 'it', 'deals', 'crushing', 'blows', 'to', 'hunger', '.', '**', 'Edit', ':', 'Living', 'on', 'campus', 'at', 'Clarkson', 'University', ',', 'I', 'have', 'had', 'food', 'delivered', 'before', '.', 'This', 'was', 'back', 'between', "'05", 'and', "'09", 'and', 'I', 'do', "n't", 'remember', 'how', 'many', 'times', 'we', "'ve", 'had', 'it', 'delivered', '.', 'Perhaps', 'they', 'do', "n't", 'deliver', 'anymore', ',', 'but', 'the', 'deliciousness', 'of', 'a', 'mezza', 'luna', 'certainly', 'warrants', 'a', 'pickup', '.']
    gold_heads = [1, 1, 1, 4, 5, 1, 8, 8, 1, 10, 8, 12, 8, 1, 15, 15, 15, 26, 19, 17, 22, 22, 17, 26, 26, 26, 15, 28, 26, 26, 15, 35, 35, 35, 35, 35, 37, 35, 42, 42, 42, 42, 35, 44, 45, 48, 48, 48, 42, 50, 48, 35, 56, 56, 56, 56, 56, 56, 67, 67, 61, 67, 65, 65, 65, 61, 67, 56, 69, 67, 56]
    doc = Doc(Vocab(), words=words)
    gold = GoldParse(doc, words=gold_words, heads=gold_heads)
