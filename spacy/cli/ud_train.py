'''Train for CONLL 2017 UD treebank evaluation. Takes .conllu files, writes
.conllu format for development data, allowing the official scorer to be used.
'''
from __future__ import unicode_literals
import plac
import tqdm
from pathlib import Path
import re
import sys
import json

import spacy
import spacy.util
from ..tokens import Token, Doc
from ..gold import GoldParse
from ..util import compounding, minibatch_by_words
from ..syntax.nonproj import projectivize
from ..matcher import Matcher
from ..morphology import Fused_begin, Fused_inside
from .. import displacy
from collections import defaultdict, Counter
from timeit import default_timer as timer

import itertools
import random
import numpy.random
import cytoolz

from . import conll17_ud_eval

from .. import lang
from .. import lang
from ..lang import zh
from ..lang import ja
from ..lang import ru


################
# Data reading #
################

space_re = re.compile('\s+')
def split_text(text):
    return [space_re.sub(' ', par.strip()) for par in text.split('\n\n')]
 

def read_data(nlp, conllu_file, text_file, raw_text=True, oracle_segments=False,
              max_doc_length=None, limit=None):
    '''Read the CONLLU format into (Doc, GoldParse) tuples. If raw_text=True,
    include Doc objects created using nlp.make_doc and then aligned against
    the gold-standard sequences. If oracle_segments=True, include Doc objects
    created from the gold-standard segments. At least one must be True.'''
    if not raw_text and not oracle_segments:
        raise ValueError("At least one of raw_text or oracle_segments must be True")
    paragraphs = split_text(text_file.read())
    conllu = read_conllu(conllu_file)
    # sd is spacy doc; cd is conllu doc
    # cs is conllu sent, ct is conllu token
    docs = []
    golds = []
    for doc_id, (text, cd) in enumerate(zip(paragraphs, conllu)):
        sent_annots = []
        if max_doc_length is not None:
            # Draw next doc length, and randomize it (but keep expectation)
            next_max_doc_length = int(next(max_doc_length)) * random.random() * 2
            next_max_doc_length = max(1, next_max_doc_length)
        else:
            next_max_doc_length = None
        for cs in cd:
            sent = defaultdict(list)
            fused_ids = set()
            fused_orths = []
            inside_fused = False
            fused_key = None
            for id_, word, lemma, pos, tag, morph, head, dep, _, space_after in cs:
                if '.' in id_:
                    continue
                if '-' in id_:
                    if inside_fused:
                        # End fused region
                        sent['words'].extend(guess_fused_orths(fused_key, fused_orths))
                        fused_ids = set()
                        fused_orths = []
                    # Begin fused region
                    inside_fused = True
                    fuse_start, fuse_end = id_.split('-')
                    for sub_id in range(int(fuse_start), int(fuse_end)+1):
                        fused_ids.add(str(sub_id))
                    sent['tokens'].append(word)
                    fused_key = word
                    fused_orths = []
                    continue
                if id_ in fused_ids:
                    fused_orths.append(word)
                    if id_ == fuse_end and space_after != 'SpaceAfter=No':
                        sent['tokens'][-1] += ' '
                else:
                    if inside_fused:
                        # End fused region
                        sent['words'].extend(guess_fused_orths(fused_key, fused_orths))
                    fused_orths = []
                    fused_ids = set()
                    inside_fused = False
                    sent['tokens'].append(word)
                    sent['words'].append(word)
                    if space_after == '_':
                        sent['tokens'][-1] += ' '
                id_ = int(id_)-1
                head = int(head)-1 if head != '0' else id_
                sent['spaces'].append(space_after != 'SpaceAfter=No')
                sent['tags'].append(tag)
                sent['heads'].append(head)
                sent['deps'].append('ROOT' if dep == 'root' else dep)
            if inside_fused:
                sent['words'].extend(guess_fused_orths(fused_key, fused_orths))
            assert len(sent['words']) == len(sent['spaces'])
            sent['entities'] = ['-'] * len(sent['words'])
            sent['heads'], sent['deps'] = projectivize(sent['heads'],
                                                       sent['deps'])
            if oracle_segments:
                docs.append(Doc(nlp.vocab, words=sent['words'], spaces=sent['spaces']))
                golds.append(GoldParse(docs[-1], words=sent['words'],
                                       heads=sent['heads'],
                                       tags=sent['tags'], deps=sent['deps'],
                                       entities=sent['entities']))

            sent_annots.append(sent)
            if raw_text and next_max_doc_length and len(sent_annots) >= next_max_doc_length:
                doc, gold = _make_gold(nlp, None, sent_annots)
                sent_annots = []
                docs.append(doc)
                golds.append(gold)
                if limit and len(docs) >= limit:
                    return docs, golds

        if raw_text and sent_annots:
            doc, gold = _make_gold(nlp, None, sent_annots)
            docs.append(doc)
            golds.append(gold)
        if limit and len(docs) >= limit:
            return docs, golds
    return docs, golds


def read_conllu(file_):
    docs = []
    sent = []
    doc = []
    for line in file_:
        if line.startswith('# newdoc'):
            if doc:
                docs.append(doc)
            doc = []
        elif line.startswith('#'):
            continue
        elif not line.strip():
            if sent:
                doc.append(sent)
            sent = []
        else:
            sent.append(list(line.strip().split('\t')))
            if len(sent[-1]) != 10:
                print(repr(line))
                raise ValueError
    if sent:
        doc.append(sent)
    if doc:
        docs.append(doc)
    return docs


def _make_gold(nlp, text, sent_annots):
    # Flatten the conll annotations, and adjust the head indices
    flat = defaultdict(list)
    for sent in sent_annots:
        flat['heads'].extend(len(flat['words'])+head for head in sent['heads'])
        for field in ['words', 'tags', 'deps', 'entities', 'tokens']:
            flat[field].extend(sent[field])
    if text is None:
        text = ''.join(flat['tokens'])
    doc = nlp.make_doc(text)
    flat.pop('tokens')
    gold = GoldParse(doc, **flat)
    return doc, gold

#############################
# Data transforms for spaCy #
#############################

def golds_to_gold_tuples(docs, golds):
    '''Get out the annoying 'tuples' format used by begin_training, given the
    GoldParse objects.'''
    tuples = []
    for doc, gold in zip(docs, golds):
        text = doc.text
        ids, words, tags, heads, labels, iob = zip(*gold.orig_annot)
        sents = [((ids, words, tags, heads, labels, iob), [])]
        tuples.append((text, sents))
    return tuples


##############
# Evaluation #
##############

def evaluate(nlp, text_loc, gold_loc, sys_loc, limit=None, oracle_segments=False):
    if oracle_segments:
        docs = []
        with gold_loc.open() as file_:
            for conllu_doc in read_conllu(file_):
                for conllu_sent in conllu_doc:
                    words = [line[1] for line in conllu_sent]
                    docs.append(Doc(nlp.vocab, words=words))
        for name, component in nlp.pipeline:
            docs = list(component.pipe(docs))
    else:
        with text_loc.open('r', encoding='utf8') as text_file:
            texts = split_text(text_file.read())
            docs = list(nlp.pipe(texts))
    with sys_loc.open('w', encoding='utf8') as out_file:
        success = write_conllu(docs, out_file)
    if not success:
        return None, None
    with gold_loc.open('r', encoding='utf8') as gold_file:
        gold_ud = conll17_ud_eval.load_conllu(gold_file)
        with sys_loc.open('r', encoding='utf8') as sys_file:
            sys_ud = conll17_ud_eval.load_conllu(sys_file)
        scores = conll17_ud_eval.evaluate(gold_ud, sys_ud)
    return docs, scores


def write_conllu(docs, file_):
    merger = Matcher(docs[0].vocab)
    merger.add('SUBTOK', None, [{'DEP': 'subtok', 'op': '+'}])
    for i, doc in enumerate(docs):
        matches = merger(doc)
        spans = [doc[start:end+1] for _, start, end in matches]
        offsets = [(span.start_char, span.end_char) for span in spans]
        for start_char, end_char in offsets:
            doc.merge(start_char, end_char)
        # TODO: This shuldn't be necessary? Should be handled in merge
        for word in doc:
            if word.i == word.head.i:
                word.dep_ = 'ROOT'
        file_.write("# newdoc id = {i}\n".format(i=i))
        for j, sent in enumerate(doc.sents):
            file_.write("# sent_id = {i}.{j}\n".format(i=i, j=j))
            file_.write("# text = {text}\n".format(text=sent.text))
            for k, token in enumerate(sent):
                file_.write(_get_token_conllu(token, k, len(sent)) + '\n')
            file_.write('\n')
            for word in sent:
                if word.head.i == word.i and word.dep_ == 'ROOT':
                    break
            else:
                #print("Rootless sentence!")
                #print(sent)
                #print(i)
                #for w in sent:
                #    print(w.i, w.text, w.head.text, w.head.i, w.dep_)
                #raise ValueError
                return False
    return True

def _get_token_conllu(token, k, sent_len):
    if token.check_morph(Fused_begin) and (k+1 < sent_len):
        n = 1
        text = [token.text]
        while token.nbor(n).check_morph(Fused_inside):
            text.append(token.nbor(n).text)
            n += 1
        id_ = '%d-%d' % (k+1, (k+n))
        fields = [id_, ''.join(text)] + ['_'] * 8
        lines = ['\t'.join(fields)]
    else:
        lines = []
    if token.head.i == token.i:
        head = 0
    else:
        head = k + (token.head.i - token.i) + 1
    fields = [str(k+1), token.text, token.lemma_, token.pos_, token.tag_, '_',
              str(head), token.dep_.lower(), '_', '_']
    if token.check_morph(Fused_begin) and (k+1 < sent_len):
        if k == 0:
            fields[1] = token.norm_[0].upper() + token.norm_[1:]
        else:
            fields[1] = token.norm_
    elif token.check_morph(Fused_inside):
        fields[1] = token.norm_
    elif token._.split_start is not None:
        split_start = token._.split_start
        split_end = token._.split_end
        split_len = (split_end.i - split_start.i) + 1
        n_in_split = token.i - split_start.i
        subtokens = guess_fused_orths(split_start.text, [''] * split_len)
        fields[1] = subtokens[n_in_split]

    lines.append('\t'.join(fields))
    return '\n'.join(lines)


def print_progress(itn, losses, ud_scores):
    fields = {
        'dep_loss': losses.get('parser', 0.0),
        'tag_loss': losses.get('tagger', 0.0),
    }
    if ud_scores is not None:
        fields.update({
            'words': ud_scores['Words'].f1 * 100,
            'sents': ud_scores['Sentences'].f1 * 100,
            'tags': ud_scores['XPOS'].f1 * 100,
            'uas': ud_scores['UAS'].f1 * 100,
            'las': ud_scores['LAS'].f1 * 100,
        })
    else:
        fields.update({
            'words': 0.0,
            'sents': 0.0,
            'tags': 0.0,
            'uas': 0.0,
            'las': 0.0
        })
    header = ['Epoch', 'Loss', 'LAS', 'UAS', 'TAG', 'SENT', 'WORD']
    if itn == 0:
        print('\t'.join(header))
    tpl = '\t'.join((
        '{:d}',
        '{dep_loss:.1f}',
        '{las:.1f}',
        '{uas:.1f}',
        '{tags:.1f}',
        '{sents:.1f}',
        '{words:.1f}',
    ))
    print(tpl.format(itn, **fields))
    return fields

#def get_sent_conllu(sent, sent_id):
#    lines = ["# sent_id = {sent_id}".format(sent_id=sent_id)]

def get_token_conllu(token, i):
    if token._.begins_fused:
        n = 1
        while token.nbor(n)._.inside_fused:
            n += 1
        id_ = '%d-%d' % (i, i+n)
        lines = [id_, token.text, '_', '_', '_', '_', '_', '_', '_', '_']
    else:
        lines = []
    if token.head.i == token.i:
        head = 0
    else:
        head = i + (token.head.i - token.i) + 1
    split_start = token._.split_start
    if split_start is not None:
        split_end = token._.split_end
        split_len = split_end.i - split_start.i
        n_in_split = token.i - split_start.i
        text_len = len(split_start.text)
        assert text_len > split_len
        if n_in_split == 0:
            text = token.text[:text_len - split_len]
        else:
            start = (text_len - split_len) + (n_in_split-1)
            end = start + 1
            text = split_start.text[start : end]
    else:
        text = token.text

    fields = [str(i+1), text, token.lemma_, token.pos_, token.tag_, '_',
              str(head), token.dep_.lower(), '_', '_']
    lines.append('\t'.join(fields))
    return '\n'.join(lines)


def get_token_split_start(token):
    if token.text == '':
        assert token.i != 0
        i = -1
        while token.nbor(i).text == '':
            i -= 1
        return token.nbor(i)
    elif (token.i+1) < len(token.doc) and token.nbor(1).text == '':
        return token
    else:
        return None


def get_token_split_end(token):
    if (token.i+1) == len(token.doc):
        return token if token.text == '' else None
    elif token.text != '' and token.nbor(1).text != '':
        return None
    i = 1
    while (token.i+i) < len(token.doc) and token.nbor(i).text == '':
        i += 1
    return token.nbor(i-1)
 

Token.set_extension('get_conllu_lines', method=get_token_conllu)
Token.set_extension('split_start', getter=get_token_split_start)
Token.set_extension('split_end', getter=get_token_split_end)
Token.set_extension('begins_fused', default=False)
Token.set_extension('inside_fused', default=False)


##################
# Initialization #
##################


def load_nlp(corpus, config, vectors=None):
    lang = corpus.split('_')[0]
    nlp = spacy.blank(lang)
    if config.vectors:
        if not vectors:
            raise ValueError("config asks for vectors, but no vectors "
                             "directory set on command line (use -v)")
        nlp.vocab.from_disk(Path(vectors) / corpus / 'vocab')
    nlp.meta['treebank'] = corpus
    return nlp

def initialize_pipeline(nlp, docs, golds, config, device):
    nlp.add_pipe(nlp.create_pipe('parser'))
    if config.multitask_tag:
        nlp.parser.add_multitask_objective('tag')
    if config.multitask_sent:
        nlp.parser.add_multitask_objective('sent_start')
    if config.multitask_vectors:
        assert nlp.vocab.vectors.size
        nlp.parser.add_multitask_objective('vectors')
    nlp.add_pipe(nlp.create_pipe('tagger'))
    for gold in golds:
        for i, tag in enumerate(gold.tags):
            if isinstance(tag, list):
                for subtag in tag:
                    if isinstance(subtag, tuple):
                        subtag = subtag[0]
                    nlp.tagger.add_label(subtag)
            else:
                if tag is not None:
                    if isinstance(tag, tuple):
                        tag = tag[0]
                    nlp.tagger.add_label(tag)
    return nlp.begin_training(lambda: golds_to_gold_tuples(docs, golds), device=device)


def extract_tokenizer_exceptions(paths, min_freq=20):
    with paths.train.conllu.open() as file_:
        conllu = read_conllu(file_)
    fused = defaultdict(lambda: defaultdict(list))
    for doc in conllu:
        for sent in doc:
            for i, token in enumerate(sent):
                if '-' in token[0]:
                    start, end = token[0].split('-')
                    length = int(end) - int(start)
                    if length < len(token[1]):
                        subtokens = sent[i+1 : i+1+length+1]
                        forms = [t[1].lower() for t in subtokens]
                        fused[token[1]][tuple(forms)].append(subtokens)
    exc = {}
    all_exceptions = []
    for word, expansions in fused.items():
        by_freq = [(len(occurs), key, occurs) for key, occurs in expansions.items()]
        by_freq.sort(reverse=True)
        for freq, subtoken_norms, occurs in by_freq:
            all_exceptions.append((freq, word, subtoken_norms))
        freq, subtoken_norms, occurs = max(by_freq)
        if freq < min_freq:
            continue
        subtoken_orths = guess_fused_orths(word, subtoken_norms)
        analysis = []
        for orth, norm in zip(subtoken_orths, subtoken_norms):
            assert len(orth) != 0, (word, subtoken_orths)
            assert len(norm) != 0
            analysis.append({'ORTH': orth, 'NORM': norm})
        analysis[0]['morphology'] = [Fused_begin]
        for subtoken in analysis[1:]:
            subtoken['morphology'] = [Fused_inside]
            subtoken['SENT_START'] = -1
        exc[word] = analysis
    all_exceptions.sort(reverse=True)
    return exc


def guess_fused_orths(word, ud_forms):
    '''The UD data 'fused tokens' don't necessarily expand to keys that match
    the form. We need orths that exact match the string. Here we make a best
    effort to divide up the word.'''
    if word == ''.join(ud_forms):
        # Happy case: we get a perfect split, with each letter accounted for.
        return ud_forms
    elif len(word) == sum(len(subtoken) for subtoken in ud_forms):
        # Unideal, but at least lengths match.
        output = []
        remain = word
        for subtoken in ud_forms:
            assert len(subtoken) >= 1
            output.append(remain[:len(subtoken)])
            remain = remain[len(subtoken):]
        assert len(remain) == 0, (word, ud_forms, remain)
        return output
    else:
        # Let's say word is 6 long, and there are three subtokens. The orths
        # *must* equal the original string. Arbitrarily, split [4, 1, 1]
        first = word[:len(word)-(len(ud_forms)-1)]
        output = [first]
        remain = word[len(first):]
        for i in range(1, len(ud_forms)):
            #assert remain, (word, ud_forms)
            output.append(remain[:1])
            remain = remain[1:]
        assert len(remain) == 0, (word, output, remain)
        return output


########################
# Command line helpers #
########################

class Config(object):
    def __init__(self, vectors=None, max_doc_length=10, multitask_tag=True,
            multitask_sent=True, multitask_vectors=False,
            nr_epoch=30, batch_size=1000, dropout=0.2):
        for key, value in locals().items():
            setattr(self, key, value)

    @classmethod
    def load(cls, loc):
        with Path(loc).open('r', encoding='utf8') as file_:
            cfg = json.load(file_)
        return cls(**cfg)


class Dataset(object):
    def __init__(self, path, section):
        self.path = path
        self.section = section
        self.conllu = None
        self.text = None
        for file_path in self.path.iterdir():
            name = file_path.parts[-1]
            if section in name and name.endswith('conllu'):
                self.conllu = file_path
            elif section in name and name.endswith('txt'):
                self.text = file_path
        if self.conllu is None:
            msg = "Could not find .txt file in {path} for {section}"
            raise IOError(msg.format(section=section, path=path))
        if self.text is None:
            msg = "Could not find .txt file in {path} for {section}"
        self.lang = self.conllu.parts[-1].split('-')[0].split('_')[0]
        self.treebank = self.conllu.parts[-1].split('-')[0]


class TreebankPaths(object):
    def __init__(self, ud_path, treebank, **cfg):
        self.train = Dataset(ud_path / treebank, 'train')
        self.dev = Dataset(ud_path / treebank, 'dev')
        self.lang = self.train.lang
        self.treebank = self.train.treebank


@plac.annotations(
    ud_dir=("Path to Universal Dependencies corpus", "positional", None, Path),
    corpus=("UD corpus to train and evaluate on, e.g. en, es_ancora, etc",
            "positional", None, str),
    output_dir=("Directory to write the development parses", "positional", None, Path),
    config=("Path to json formatted config file", "positional"),
    limit=("Size limit", "option", "n", int),
    use_gpu=("Use GPU", "option", "g", int),
    use_oracle_segments=("Use oracle segments", "flag", "G", int),
    vectors_dir=("Path to directory with pre-trained vectors, named e.g. en/",
                 "option", "v", Path),
)
def main(ud_dir, output_dir, config, corpus, vectors_dir=None,
         limit=0, use_gpu=-1, use_oracle_segments=False):
    spacy.util.fix_random_seed()
    lang.zh.Chinese.Defaults.use_jieba = False
    lang.ja.Japanese.Defaults.use_janome = False
    lang.ru.Russian.Defaults.use_pymorphy2 = False

    config = Config.load(config)
    paths = TreebankPaths(ud_dir, corpus)
    model_output = output_dir / corpus / 'best-model'
    if not (output_dir / corpus).exists():
        (output_dir / corpus).mkdir()
    if not model_output.exists():
        model_output.mkdir()
    print("Train and evaluate", corpus, "using lang", paths.lang)
    nlp = load_nlp(paths.treebank, config, vectors=vectors_dir)
    tokenizer_exceptions = extract_tokenizer_exceptions(paths)
    for orth, subtokens in tokenizer_exceptions.items():
        nlp.tokenizer.add_special_case(orth, subtokens)
    
    docs, golds = read_data(nlp, paths.train.conllu.open(), paths.train.text.open(),
                            max_doc_length=None, limit=limit)

    optimizer = initialize_pipeline(nlp, docs, golds, config, use_gpu)

    batch_sizes = compounding(config.batch_size/10, config.batch_size, 1.001)
    max_doc_length = compounding(5., 20., 1.001)
 
    best_score = 0.0
    training_log = []
    for i in range(config.nr_epoch):
        docs, golds = read_data(nlp, paths.train.conllu.open(), paths.train.text.open(),
                                max_doc_length=max_doc_length, limit=limit,
                                oracle_segments=use_oracle_segments,
                                raw_text=not use_oracle_segments)
        Xs = list(zip(docs, golds))
        random.shuffle(Xs)
        batches = minibatch_by_words(Xs, size=batch_sizes)
        losses = {}
        n_train_words = sum(len(doc) for doc in docs)
        with tqdm.tqdm(total=n_train_words, leave=False) as pbar:
            for batch in batches:
                if not batch:
                    continue
                batch_docs, batch_gold = zip(*batch)
                batch_docs = list(batch_docs)
                batch_gold = list(batch_gold)
                pbar.update(sum(len(doc) for doc in batch_docs))
                nlp.update(batch_docs, batch_gold, sgd=optimizer,
                           drop=config.dropout, losses=losses)
        
        parses_path = output_dir / corpus / 'epoch-{i}.conllu'.format(i=i)
        with nlp.use_params(optimizer.averages):
            try:
                parsed_docs, dev_scores = evaluate(nlp, paths.dev.text, paths.dev.conllu, parses_path,
                                                   oracle_segments=use_oracle_segments)
            except RecursionError:
                dev_scores = None
                parsed_docs = None
            except IndexError:
                dev_scores = None
                parsed_docs = None
            training_log.append(print_progress(i, losses, dev_scores))
            if parsed_docs is not None:
                _render_parses(i, parsed_docs[:50]) 
            if dev_scores is not None and dev_scores['LAS'].f1 >= best_score:
                nlp.meta['log'] = training_log
                nlp.to_disk(model_output)
                best_score = dev_scores['LAS'].f1
            else:
                optimizer.alpha *= 0.5


def _render_parses(i, to_render):
    to_render[0].user_data['title'] = "Batch %d" % i
    with Path('/tmp/parses.html').open('w') as file_:
        html = displacy.render(to_render[:5], style='dep', page=True)
        file_.write(html)


if __name__ == '__main__':
    plac.call(main)
