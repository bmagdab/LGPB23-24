import stanza
import re
import syll
import csv
import pandas as pd
from tqdm import tqdm
import torch
import argparse
import os
from datetime import datetime
from stanza.utils.conll import CoNLL
from collections import deque


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-f', nargs='+')
arg_parser.add_argument('-d', action='store_true')
arg_parser.add_argument('-s', action='store_true')
arg_parser.add_argument('-t', action='store_true')
args = arg_parser.parse_args()

if args.s and args.t:
    print('cant have em both!')
elif not (args.s or args.t):
    parsing = True
    nlp = stanza.Pipeline(lang='en', use_gpu=True, processors='tokenize, lemma, pos, depparse, ner',
                          download_method=stanza.DownloadMethod.REUSE_RESOURCES, tokenize_no_ssplit=True)
    nlpcpu = stanza.Pipeline(lang='en', use_gpu=False, processors='tokenize, lemma, pos, depparse, ner',
                             download_method=stanza.DownloadMethod.REUSE_RESOURCES, tokenize_no_ssplit=True)
else:
    parsing = False


# preprocessing --------------------------------------------------------------------------------------------------------
def chunker(src):
    """
    puts the sentences from a given .tsv file into bigger chunks of text, every chunk is placed in a dictionary and is
    accessible with the @@ marker as key
    :param src: name of the source .tsv file
    :return: dict with the text chunks, dict with sent_ids, genre, year and source of the parsed texts given in the .tsv
    """
    path = os.getcwd() + '/inp/' + src
    texts = {} # dictionary for texts to parse
    sent_ids = {} # dictionary for ids of parsed sentences
    df = pd.read_csv(path, sep='\t', quoting=csv.QUOTE_NONE, lineterminator='\n', quotechar='"')

    # przerobione od Adama
    df.dropna()
    filtered1 = df[df["SENT"].str.contains("TOOLONG") == False]
    filtered2 = filtered1[filtered1["SENT"].str.match(r"^ *\d[\d ]*$") == False]
    filtered2.to_csv(path, sep="\t", quoting=csv.QUOTE_NONE, lineterminator="\n", index=None)
    # ------

    df = pd.read_csv(path, sep='\t', quoting=csv.QUOTE_NONE, lineterminator='\n', quotechar='"')
    txt = ''
    id_list = []
    marker = ''
    genre = df.loc[1][2]
    try:
        year = str(int(df.loc[1][3]))
    except ValueError:
        # tsv files for web22-34 have something wrong in the year column, so if anything is wrong try to look for the
        # year in the sourcefile name in the next column
        year = str(int(re.search('\d+', df.loc[1][4]).group()))

    for x in tqdm(range(len(df))):
        if marker != re.match('@@[0-9]+', str(df.loc[x][5])).group() and marker != '':
            # if there's a new @@ marker, a new entry in the dictionary is created
            m = re.match('@@[0-9]+', str(marker))

            texts[m.group()] = txt
            txt = clean(str(df.loc[x][1])) + '\n\n'

            sent_ids[m.group()] = deque(reversed(id_list))
            id_list = [int(df.loc[x][0])]

            marker = re.match('@@[0-9]+', str(df.loc[x][5])).group()

        elif marker != re.match('@@[0-9]+', str(df.loc[x][5])).group():
            # if this is the first marker in the file, it's just written down for later
            marker = re.match('@@[0-9]+', str(df.loc[x][5])).group()
            txt += clean(str(df.loc[x][1])) + '\n\n'
            id_list.append(int(df.loc[x][0]))

        else:
            txt += clean(str(df.loc[x][1])) + '\n\n'
            id_list.append(int(df.loc[x][0]))

    m = re.match('@@[0-9]+', str(marker))
    texts[m.group()] = txt
    sent_ids[m.group()] = deque(reversed(id_list))
    return texts, sent_ids, genre, year


def clean(txt):
    """
    removes redundant spaces by the punctuation
    :param txt: text to clean
    :return: clean text
    """
    to_remove = []

    # this loop finds the redundant spaces and adds them to the to_remove list
    for i in range(len(txt)):
        if txt[i] == ' ':
            if i+1 == len(txt) or \
                    (len(txt) > i+3 and txt[i+1:i+3] == '...' and txt[i-1] != '.') or \
                    (txt[i + 1] in [',', '.', '!', '?', ')', '}', ']', ':', ';'] and ((i + 2 < len(txt) and txt[i + 2] == ' ') or i + 2 == len(txt))) or \
                    (txt[i - 1] in ['(', '{', '['] and txt[i - 2] == ' ') or \
                    (txt[i + 1:i + 4].casefold() == "n't".casefold()):
                if txt[i+1] == '.' and txt[i-3:i-1] == '...':
                    continue # please may this work
                to_remove.append(i)
        elif len(txt) == 1:
            break
        elif txt[i] in [',', '.', '!', '?', ')', '}', ']', '(', '{', '[', ':', ';'] and i == 0 and (len(txt) == 1 or txt[1] == ' '):
            to_remove.append(i+1)
        elif txt[i] == "'" and txt[i - 1] == ' ' and (i+1 == len(txt) or txt[i + 1] != ' '):
            to_remove.append(i-1)
    # there was an elif for " too, but I should've removed it a long time ago I think, maybe I should do the same for '?

    # the ellipsis will be the death of me...
    if '... .' in txt:
        ellipses = re.finditer(re.escape('... .'), txt)
        for e in ellipses:
            if e.start() + 3 in to_remove:
                to_remove.remove(e.start() + 3)
                # '... .'[3] returns the space, I have to get those specific spaces out of this list
    if '. ...' in txt:
        ellipses = re.finditer(re.escape('. ...'), txt)
        for e in ellipses:
            if e.start() + 1 in to_remove:
                to_remove.remove(e.start() + 1)

    to_remove.sort()
    to_remove.reverse()
    # removes the spaces marked by indices in the list
    for i in to_remove:
        if txt[i] == ' ':
            txt = txt[:i] + txt[i+1:]
    return txt


def get_info_from_conll(sentence):
    """
    gets the sentence text and sentence id for sentences in conllu files from trankit
    :param sentence: sentence object from a stanza doc
    :return:
    """
    found = 0 # I need both the sentence text and the right sentence id, this will keep track
    take_previous_textid = False
    for comment in sentence.comments:
        if '# SENTENCE : ' in comment:
            sentence.text = comment[len('# SENTENCE : '):]
            found += 1
            continue

        if '# ID : ' in comment:
            # sometimes in the ID field there's something like @@134545-987 which is fine, but then other times there's
            # @@134545TOOLONG blablah once upon-987 or @@134545-987.0 or even @@134545TOOLONG blablah once upon-987.0

            # also there can be a NA in the ID field
            if re.search('@@[0-9]{1,8}', comment):
                textid = re.search('@@[0-9]{1,8}', comment).group()
            elif re.search('[0-9]{1,8}$', sentence.text):
                partoftextid = re.search('[0-9]{1,8}$', sentence.text)
                textid = '@@' + partoftextid.group()
            else:
                textid = None

            if comment[-2:] == '.0' and re.search('-[0-9]+$', comment[:-2]):
                sentid = re.search('-[0-9]+$', comment[:-2]).group()
            elif re.search('-[0-9]+$', comment):
                sentid = re.search('-[0-9]+$', comment).group()
            else:
                sentid = '-nan'

            if not textid:
                take_previous_textid = sentid
            else:
                sentence.sent_id = textid + sentid
            found += 1
            continue

    # a sent_id is given for every sentence, so it can be used to identify where an error occurred, but this is not the
    # id that I need for the csv table later
    assert found == 2, 'this conllu lacks information: ' + sentence.sent_id
    return take_previous_textid


# working with sentences -----------------------------------------------------------------------------------------------
def dep_children(sentence):
    """
    for every word in a given sentence finds a list of IDs of that words dependents
    :param sentence: a Stanza object
    :return: nothing
    """
    for word in sentence.words:
        word.children = []
        for w in sentence.words:
            if w.head == word.id:
                word.children.append(w.id)


def word_indexer(sentence):
    """
    adds information about the starting and ending characters of a word; needed for trankit conllu files and for when
    there are double spaces in sentences in tsv files
    :param sentence: stanza sentence object
    :return:
    """
    sent_text = sentence.text
    current_id = 0
    for word in sentence.words:
        word_area = sent_text[:len(word.text)*2]
        # to not search the whole sentence, because if there are some inaccuracies with spaces in a token, it might match
        # something further in the sentence

        if re.search(re.escape(word.text), word_area):
            match = re.search(re.escape(word.text), word_area)
        elif "'" in word.text and '\\' in word.text and re.search(word.text, word_area):
            # this should match a token "\'", which is not matched by re.escape()
            match = re.search(word.text, word_area)
        elif re.search(re.escape(re.sub(' +', ' ', word.text)), word_area):
            # maybe 1st degree cleaning will help...
            word.text = re.sub(' +', ' ', word.text)
            match = re.search(re.escape(word.text), word_area)
        elif re.search(re.escape(clean(re.sub(' +', ' ', word.text))), word_area):
            # ... or the 2nd degree cleaning!
            word.text = re.sub(' +', ' ', word.text)
            word.text = clean(word.text)
            match = re.search(re.escape(word.text), word_area)
        else:
            # so there was this one case of a smiley face and the clean() function applied to the whole sentence
            # changed it from : - ) to : -) and it could not match, so I decided to say it matches for any number of
            # spaces anywhere in the expression
            word_temp = word.text.replace('*', '\*').replace(' ', '').replace('', '\\s*').replace('**', '*\*').replace(')', '\\)').replace('(', '\\(').replace('?', '\?').replace('.', '\.').replace('+', '\+').replace('$', '\$')
            # above I have to add expressions to escape by hand, because I don't want it to escape the \\s*
            # I know this looks horrible, but I haven't come up with any other way to do it
            match = re.search(word_temp, word_area)

        try:
            word.start = match.start() + current_id
        except AttributeError:
            print(sentence.sent_id)
        word.end = match.end() + current_id
        sent_text = sent_text[match.end():]
        current_id = match.end() + current_id


def coord_info(crd, sent, conj, other_ids):
    """
    collects information on elements of a coordination: text of the conjunct, number of words, tokens and syllables
    :param crd: coordination (a dictionary containing the left and right elements of coordination, its head and the conjunct, if there is one)
    :param sent: sentence (Stanza object) where the coordination was found
    :param conj: which element of a coordination is to be considered (takes values "L" or "R")
    :param other_ids: a list of word IDs, that belong to the opposite element of the coordination
    :return: a list of word IDs, that belong to the element of the coordination that was specified by the conj parameter
    """
    txt_ids = []

    for id in crd[conj].children:
        if conj == 'L' \
                and (id < min(crd['other_conjuncts']) if crd['other_conjuncts'] else
                    (id < crd['conj'].id if 'conj' in crd.keys() else
                    (id < min(other_ids) if other_ids else True))) \
                and id not in crd['other_conjuncts'] \
                and id != crd['R'].id:
            # if we're looking at the left conjunct, we only take into account dependecies of the conjuncts head if they
            # appear before any other elements of the coordination (this matters only if we use incorrectly parsed sentences
            # and unfortunately stanza sometimes returns those)

            if id < crd['L'].id and re.match('compound', sent.words[id - 1].deprel):
                # compound dependencies of the head appearing on the left of the head are always part of the conjunct
                txt_ids.append(id)
                continue

            elif id < crd['L'].id:
                # other dependencies appearing on the left of the head are not part of the conjunct, but should be
                # considered dependencies of the whole coordination, if other conjuncts do not have a dependency with
                # the same label

                for c in crd['other_conjuncts']+[crd['R'].id]:
                    for c_child in sent.words[c - 1].children:
                        if sent.words[id - 1].deprel != 'cc' and (sent.words[c_child - 1].deprel == sent.words[id - 1].deprel
                                or (re.match('subj', sent.words[c_child - 1].deprel) and re.match('subj', sent.words[id - 1].deprel))
                                or (re.match('nmod', sent.words[c_child - 1].deprel) and re.match('nmod', sent.words[id - 1].deprel))):
                            # in this case, the subj labels and its subcategories should be treated equally, same with nmod
                            txt_ids.append(id)
                            break
                    if id in txt_ids:
                        break
            else:
                # all children of the head appearing between the head and other elements of the coordination are part of the conjunct
                txt_ids.append(id)
        elif conj == 'R' and ('conj' not in crd.keys() or id != crd['conj'].id):
            # all children of the right conjuncts head should be part of the coordination (besides the conjunction)
            txt_ids.append(id)

    # the loop below looks through words already included in the conjunct and adds their children, repeats until there
    # are no more words to be added
    keep_looking = True
    while keep_looking and txt_ids:
        for id in txt_ids:
            for i in sent.words[id - 1].children:
                txt_ids.append(i)
            else:
                keep_looking = False
    txt_ids.append(crd[conj].id)
    txt_ids.sort()

    # removes some of the punctuation from the beginning of the conjunct
    while sent.words[min(txt_ids)-1].text in [',', ';', '-', ':', '--']:
        txt_ids.remove(min(txt_ids))

    words = 0
    tokens = 0
    for id in range(min(txt_ids), max(txt_ids) + 1):
        tokens += 1
        if sent.words[id - 1].deprel != 'punct':
            words += 1

    txt = sent.text[sent.words[min(txt_ids) - 1].start - sent.words[0].start:sent.words[max(txt_ids) - 1].end - sent.words[0].start]

    syllables = 0
    for w in txt.split():
        syllables += syll.count_word(w)

    crd[conj+'conj'] = txt
    crd[conj+'words'] = words
    crd[conj+'tokens'] = tokens
    crd[conj+'syl'] = syllables

    return txt_ids


def extract_coords(doc, marker, conll_list, sent_ids):
    """
    finds the coordinations in sentences in a given document
    :param doc: Stanza object containing parsed sentences
    :param marker: the @@ marker from COCA texts needed for sentence id
    :param conll_list: list of sentences that have coordinations and that have to be included in the conllu file created
    after parsing
    :param sent_ids: dictionary that deques of sentence ids from the tsv file
    :return: list of coordinations
    """
    coordinations = []
    for sent in tqdm(doc.sentences, disable=parsing):
        # progress bar is disabled if the data has to be parsed first, because then there is a tqdm wrapper on the
        # parsing loop

        # preparing the sentence depending on its source:
        if args.t:
            textid_missing = get_info_from_conll(sent)
            if textid_missing:
                textid = re.match('@@[0-9]{1,8}', coordinations[-1]['sent_id']).group()
                sent.sent_id = textid + textid_missing

        sent.text = re.sub(' +', ' ', sent.text)
        if parsing:
            index = sent_ids[marker].pop()
        else:
            sent.text = clean(sent.text)
        dep_children(sent)
        word_indexer(sent)

        conjs = {}
        wrong_coords = []
        # every word that has a conj dependency becomes a key in the conjs dictionary, its values are all words that are
        # connected to the key with a conj dependency
        for dep in sent.dependencies:
            if dep[1] == 'conj' \
                    and dep[0].upos != 'PUNCT' and dep[2].upos != 'PUNCT' \
                    and dep[0].text not in [',', ';', '-', ':', '--'] and dep[2].text not in [',', ';', '-', ':', '--']:
                # stanza has found some coordinations between punctuation marks, which shouldn't happen
                if dep[0].id > dep[2].id:
                    # stanza has also found a few edges with the conj label directed to the left
                    if dep[0].id in conjs.keys():
                        wrong_coords.append(dep[0].id)
                    if dep[2].id in conjs.keys():
                        wrong_coords.append(dep[2].id)
                    continue
                if dep[0].id in conjs.keys():
                    conjs[dep[0].id].append(dep[2].id)
                else:
                    conjs[dep[0].id] = [dep[2].id]

        for wrong in wrong_coords:
            try:
                del conjs[wrong]
            except KeyError:
                continue

        # this loop looks through the conj dependencies and finds what coordinations they make up
        # the crds list will consist of lists of heads of every element of a given coordination
        crds = []
        for l in conjs.keys():
            crd = [l]
            temp_coord = []  # this will be used if what the loop is looking at could be a nested coordination
            cc = None  # the conjunction of a coordination
            conjs[l].sort()
            for conj in conjs[l]:
                # an element of coordination is added either to the main coordination or to the temporary (possibly nested) one
                if cc:
                    temp_coord.append(conj)
                else:
                    crd.append(conj)

                # this loop looks for a conjunction attached to the element of a coordination
                for ch in sent.words[conj - 1].children:
                    if not cc and sent.words[ch - 1].deprel == 'cc':
                        # sets a conjunction if there wasn't any found yet
                        cc = sent.words[ch - 1].text
                    elif cc and sent.words[ch - 1].deprel == 'cc' and sent.words[ch - 1].text != cc:
                        # if there was a conjunction found already and now there's a different one, we're looking at
                        # a nested coordination
                        crds.append(crd)
                        crd = [l] + temp_coord
                        cc = sent.words[ch - 1].text
                    elif cc and sent.words[ch - 1].deprel == 'cc' and sent.words[ch - 1].text == cc:
                        # if there was a conjunction found already and the new one is the same, we can clean
                        # the temp_coord list, because so far this seems to not be a nested coordination
                        crd += temp_coord
                        temp_coord = []
            # if not temp_coord:
            #     crds.append(crd)
            # this was here earlier and I feel like it shouldn't be, but now I'm not sure
            if crd not in crds:
                crds.append(crd)

            if parsing and crds:
                # if there are any valid conj dependencies in a sentence, it will be included in the .conllu file
                # a sentence id including the @@ marker from COCA source files is assigned and will be both in the .conllu
                # and .csv file
                sent.sent_id = str(marker) + '-' + str(index)
                conll_list.append(sent)

        # this loop writes down information about every coordination based on the list of elements of a coordination
        for crd in crds:
            if len(crd) > 1:
                coord = {'L': sent.words[min(crd) - 1], 'R': sent.words[max(crd) - 1]}
                crd.pop(0)
                crd.pop(-1)
                coord['other_conjuncts'] = crd
                if coord['L'].head != 0:
                    coord['gov'] = sent.words[coord['L'].head - 1]
                for child in coord['R'].children:
                    if sent.words[child-1].deprel == 'cc':
                        coord['conj'] = sent.words[child-1]
                        break
                r_ids = coord_info(coord, sent, 'R', [])
                coord_info(coord, sent, 'L', r_ids)
                coord['sentence'] = sent.text
                coord['sent_id'] = sent.sent_id
                coordinations.append(coord)

    return coordinations


# creating files -------------------------------------------------------------------------------------------------------
def create_conllu(sent_list, genre, year):
    """
    creates a conllu file with sentences with coordinations, with every tree comes the text of the sentence and the same
    sentence id that is in the csv file
    this function is based on doc2conll from stanza.utils.conll, described as deprecated and to be removed
    :param sent_list: list of sentences (stanza object) that had a coordination in them
    :param file_path: name for the conllu file
    """
    doc_conll = []
    for sentence in sent_list:
        sent_conll = []
        for com in sentence.comments:
            if re.match('# sent_id = ', com):
                sent_conll.append('# sent_id = ' + sentence.sent_id)
            else:
                sent_conll.append(com)
        for token in sentence.tokens:
            sent_conll.extend(token.to_conll_text().split('\n'))
        doc_conll.append(sent_conll)

    path = os.getcwd() + '/outp/stanza_trees_' + str(genre) + '_' + str(year) + '.conllu'
    with open(path, mode='w', encoding='utf-8') as conll_file:
        for sent in doc_conll:
            for line in sent:
                conll_file.write(line + '\n')
            conll_file.write('\n')


def create_csv(crd_list, genre, year):
    """
    creates a .csv table where every row has information about one coordination
    :param crd_list: list of dictionaries, each dictionary represents a coordination
    :param genre: genre of the parsed file, information included in the filename and in the table
    :param year: year in which the parsed text was published, included in the filename and the table
    :param source: name of the source file
    :return: nothing
    """
    if args.t:
        parser = 'trankit'
    else:
        parser = 'stanza'

    from_file = f'text_{genre}_{year}.txt'
    path = os.getcwd() + f'/outp/{parser}_coordinations_' + str(genre) + '_' + str(year) + '.csv'

    with open(path, mode='w', newline="", encoding='utf-8-sig') as outfile:
        writer = csv.writer(outfile)
        col_names = ['governor.position', 'governor.word', 'governor.tag', 'governor.pos', 'governor.ms',
                     'conjunction.word', 'conjunction.tag', 'conjunction.pos', 'conjunction.ms', 'no.conjuncts',
                     'L.conjunct', 'L.dep.label', 'L.head.word', 'L.head.tag', 'L.head.pos', 'L.head.ms', 'L.words',
                     'L.tokens', 'L.syllables', 'L.chars', 'R.conjunct', 'R.dep.label', 'R.head.word', 'R.head.tag',
                     'R.head.pos', 'R.head.ms', 'R.words', 'R.tokens', 'R.syllables', 'R.chars', 'sentence', 'sent.id',
                     'genre', 'converted.from.file']
        writer.writerow(col_names)

        for coord in crd_list:
            writer.writerow(['0' if 'gov' not in coord.keys() else 'L' if coord['L'].id > coord['gov'].id else 'R',     # governor.position
                             str(coord['gov'].text) if 'gov' in coord.keys() else '',                                   # governor.word
                             coord['gov'].xpos if 'gov' in coord.keys() else '',                                        # governor.tag
                             coord['gov'].upos if 'gov' in coord.keys() else '',                                        # governor.pos
                             coord['gov'].feats if 'gov' in coord.keys() else '',                                       # governor.ms
                             str(coord['conj'].text) if 'conj' in coord.keys() else '',                                 # conjunction.word
                             coord['conj'].xpos if 'conj' in coord.keys() else '',                                      # conjunction.tag
                             coord['conj'].upos if 'conj' in coord.keys() else '',                                      # conjunction.pos
                             coord['conj'].feats if 'conj' in coord.keys() else '',                                     # conjunction.ms
                             int(2 + len(coord['other_conjuncts'])),                                                    # no.conjuncts
                             str(coord['Lconj']),                                                                       # L.conjunct
                             coord['L'].deprel,                                                                         # L.dep.label
                             str(coord['L'].text),                                                                      # L.head.word
                             coord['L'].xpos,                                                                           # L.head.tag
                             coord['L'].upos,                                                                           # L.head.pos
                             coord['L'].feats,                                                                          # L.head.ms
                             int(coord['Lwords']),                                                                      # L.words
                             int(coord['Ltokens']),                                                                     # L.tokens
                             int(coord['Lsyl']),                                                                        # L.syllables
                             len(coord['Lconj']),                                                                       # L.chars
                             str(coord['Rconj']),                                                                       # R.conjunct
                             coord['R'].deprel,                                                                         # R.dep.label
                             str(coord['R'].text),                                                                      # R.head.word
                             coord['R'].xpos,                                                                           # R.head.tag
                             coord['R'].upos,                                                                           # R.head.pos
                             coord['R'].feats,                                                                          # R.head.ms
                             int(coord['Rwords']),                                                                      # R.words
                             int(coord['Rtokens']),                                                                     # R.tokens
                             int(coord['Rsyl']),                                                                        # R.syllables
                             len(coord['Rconj']),                                                                       # R.chars
                             str(coord['sentence']),                                                                    # sentence
                             coord['sent_id'],                                                                          # sent.id
                             genre,                                                                                     # genre
                             from_file])                                                                                # converted.from.file


# running --------------------------------------------------------------------------------------------------------------
def run(filename):
    if args.s or args.t:
        print('processing ' + filename)
        s = datetime.now()
        doc = CoNLL.conll2doc(os.getcwd() + '/inp/' + filename)
        e = datetime.now()
        print(f'reading conll took {str(e-s)}')
        print('extracting coords...')
        crds_full_list = extract_coords(doc, '', [], [])
        genre = re.search('acad|news|fic|mag|blog|web|tvm', filename).group()
        year = re.search('[0-9]+', filename).group()
        if len(year) == 1:
            year = '0' + year
    else:
        txts, sent_ids, genre, year = chunker(filename)
        if len(year) == 1:
            year = '0' + year
        crds_full_list = []
        conll_list = []

        # extracts coordinations one chunk at a time
        for mrk in tqdm(txts.keys()):
            torch.cuda.empty_cache()
            try:
                doc = nlp(txts[mrk])
            except RuntimeError:
                doc = nlpcpu(txts[mrk])
            coordinations = extract_coords(doc, mrk, conll_list, sent_ids)
            crds_full_list += coordinations

        print('processing conllu...')
        create_conllu(conll_list, genre, year)
        print('done!')

    print('creating a csv...')
    create_csv(crds_full_list, genre, year)
    print('csv created!')


if args.d:
    for file in os.listdir(os.getcwd() + '/inp/'):
        # first check if this file has already been processed
        genre = re.search('acad|news|fic|mag|blog|web|tvm', file).group()
        year = re.search('[0-9]+', file).group()
        if len(year) == 1:
            year = '0' + year

        if args.s and f'stanza_coordinations_{genre}_{year}.csv' in os.listdir(os.getcwd() + '/outp/'):
            continue
        elif args.t and f'trankit_coordinations_{genre}_{year}.csv' in os.listdir(os.getcwd() + '/outp/'):
            continue
        else:
            run(file)
else:
    for file in args.f:
        run(file)
