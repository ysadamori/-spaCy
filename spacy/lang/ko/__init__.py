# encoding: utf8
from __future__ import unicode_literals, print_function
import re

from ...language import Language
from ...attrs import LANG
from ...tokens import Doc
from ...tokenizer import Tokenizer
from ..char_classes import LIST_PUNCT


class KoreanDefaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    lex_attr_getters[LANG] = lambda text: 'ko'
    suffixes = tuple(LIST_PUNCT) + ('\.',)


class Korean(Language):
    lang = 'ko'
    Defaults = KoreanDefaults


__all__ = ['Korean']
