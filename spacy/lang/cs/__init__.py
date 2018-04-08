# coding: utf8
from __future__ import unicode_literals

from ..tokenizer_exceptions import BASE_EXCEPTIONS
from ..norm_exceptions import BASE_NORMS
from ...language import Language
from ...attrs import LANG, NORM
from ...util import update_exc, add_lookups


class CzechDefaults(Language.Defaults):
    lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
    lex_attr_getters[LANG] = lambda text: 'cs'
    lex_attr_getters[NORM] = add_lookups(Language.Defaults.lex_attr_getters[NORM], BASE_NORMS)
    tokenizer_exceptions = dict(BASE_EXCEPTIONS)
    stop_words = []


class Czech(Language):
    lang = 'cs'
    Defaults = CzechDefaults


__all__ = ['Czech']

