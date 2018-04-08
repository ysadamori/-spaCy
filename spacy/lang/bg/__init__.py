# coding: utf8
from __future__ import unicode_literals

from ..tokenizer_exceptions import BASE_EXCEPTIONS
from ..norm_exceptions import BASE_NORMS
from ...language import Language
from ...attrs import LANG, NORM
from ...util import update_exc, add_lookups


class BulgarianDefaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    lex_attr_getters[LANG] = lambda text: 'bg'
    lex_attr_getters[NORM] = add_lookups(Language.Defaults.lex_attr_getters[NORM], BASE_NORMS)
    tokenizer_exceptions = dict(BASE_EXCEPTIONS)
    stop_words = []


class Bulgarian(Language):
    lang = 'bg'
    Defaults = BulgarianDefaults


__all__ = ['Bulgarian']

