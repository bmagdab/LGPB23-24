import stanza
import re
import syll
import csv
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
import argparse
import os
import gc
from stanza.utils.conll import CoNLL


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-f', nargs='+')
args = arg_parser.parse_args()
nlp = stanza.Pipeline(lang='en', use_gpu=True, processors='tokenize, lemma, pos, depparse, ner', download_method=stanza.DownloadMethod.REUSE_RESOURCES, tokenize_no_ssplit=True)
nlpcpu = stanza.Pipeline(lang='en', use_gpu=False, processors='tokenize, lemma, pos, depparse, ner', download_method=stanza.DownloadMethod.REUSE_RESOURCES, tokenize_no_ssplit=True)


# preparing the text ---------------------------------------------------------------------------------------------------
def chunker(src):
    """
    puts the sentences from a given .tsv file into bigger chunks of text, every chunk is placed in a dictionary and is
    accessible with the @@ marker as key
    :param src: name of the source .tsv file
    :return: dictionary with the text chunks, genre, year and source of the parsed texts given in the .tsv
    """
    texts = {}
    df = pd.read_csv(src, sep='\t', quoting=csv.QUOTE_NONE, lineterminator='\n', quotechar='"')
    # przerobione od Adama
    df.dropna()
    filtered1 = df[df["SENT"].str.contains("TOOLONG") == False]
    filtered2 = filtered1[filtered1["SENT"].str.match(r"^ *\d[\d ]*$") == False]
    filtered2.to_csv(src, sep="\t", quoting=csv.QUOTE_NONE, lineterminator="\n", index=None)
    # ------
    df = pd.read_csv(src, sep='\t', quoting=csv.QUOTE_NONE, lineterminator='\n', quotechar='"')
    txt = ''
    marker = ''
    genre = df.loc[1][2]
    try:
        year = str(int(df.loc[1][3]))
    except ValueError: # może usuń to potem albo nie idk to jest przez to że w web22-34 źle był wpisany rok
        year = src[-6:-4]
    source = df.loc[1][4]
    for x in tqdm(range(len(df))):
        if pd.isna(df.loc[x][5]) or not re.match('@@[0-9]+', str(df.loc[x][5])):
            # if there's missing information in the line, the text of the sentence is just appended
            txt += clean(str(df.loc[x][1])) + '\n\n'

        elif marker != re.match('@@[0-9]+', str(df.loc[x][5])).group() and marker != '':
            # if there's a new @@ marker, a new entry in the dictionary is created
            m = re.match('@@[0-9]+', str(marker))
            texts[m.group()] = txt
            txt = clean(str(df.loc[x][1])) + '\n\n'
            marker = re.match('@@[0-9]+', str(df.loc[x][5])).group()

        elif marker != re.match('@@[0-9]+', str(df.loc[x][5])).group():
            # if this is the first marker in the file, it's just written down for later
            marker = re.match('@@[0-9]+', str(df.loc[x][5])).group()
            txt += clean(str(df.loc[x][1])) + '\n\n'

        else:
            txt += clean(str(df.loc[x][1])) + '\n\n'
    m = re.match('@@[0-9]+', str(marker))
    texts[m.group()] = txt
    return texts, genre, year, source


def clean(txt):
    """
    removes redundant spaces by the punctuation
    :param txt: text to clean
    :return: clean text
    """
    to_remove = []
    qm = 2
    # 0 - quotation mark was closed, 1 - quotation mark was opened, 2 - quotation mark was not set

    # this loop finds the redundant spaces and adds them to the to_remove list
    for i in range(len(txt)):
        if txt[i] == ' ':
            if i+1 == len(txt) or \
                    (len(txt) > i+3 and txt[i+1:i+3] == '...') or \
                    (txt[i + 1] in [',', '.', '!', '?', ')', '}', ']', ':', ';'] and ((i + 2 < len(txt) and txt[i + 2] == ' ') or i + 2 == len(txt))) or \
                    (txt[i - 1] in ['(', '{', '['] and txt[i - 2] == ' ') or \
                    (txt[i + 1:i + 4] == "n't"):
                to_remove.append(i)
        elif len(txt) == 1:
            break
        elif txt[i] in [',', '.', '!', '?', ')', '}', ']', '(', '{', '[', ':', ';'] and i == 0 and (len(txt) == 1 or txt[1] == ' '):
            to_remove.append(i+1)
        elif txt[i] == "'" and txt[i - 1] == ' ' and (i+1 == len(txt) or txt[i + 1] != ' '):
            to_remove.append(i-1)
        elif txt[i] == '"':
            if i == 0 and txt[1] == ' ':
                to_remove.append(1)
                qm = 1
            elif i == len(txt)-1 and txt[i-1] == ' ':
                to_remove.append(i-1)
            elif txt[i - 1] == ' ' and (i + 1 == len(txt) or txt[i + 1] == ' '):
                if qm == 1:
                    to_remove.append(i - 1)
                    qm = 0
                else:
                    to_remove.append(i + 1)
                    qm = 1

    to_remove.sort()
    to_remove.reverse()
    # removes the spaces marked by indices in the list
    for i in to_remove:
        txt = txt[:i] + txt[i+1:]
    return txt


# working with stanza sentences ----------------------------------------------------------------------------------------
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
    sent_text = sentence.text
    current_id = 0
    for word in sentence.words:
        match = re.search(re.escape(word.text), sent_text)
        word.start = match.start() + current_id
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


def extract_coords(src, marker, conll_list, sentence_count):
    """
    finds coordinations in a given text, creates a conllu file containing every sentence with a found coordination
    :param src: text to parse
    :param marker: marker of the parsed text
    :param conll_list: list for sentences, to later create a conllu file corresponding to the table of coordinations
    :param sentence_count: counts how many sentences from the whole source file have been parsed
    :return: list of dictionaries representing coordinations, the number of sentences that were already processed
    """
    torch.cuda.empty_cache()
    try:
        doc = nlp(src)
    except RuntimeError:
        doc = nlpcpu(src)
    coordinations = []
    for sent in doc.sentences:
        index = sent.index + 1 + sentence_count
        # updating sentence count so that it corresponds to the sentence ids in the source .tsv file
        dep_children(sent)
        word_indexer(sent)
        sent.text = re.sub(' +', ' ', sent.text)

        conjs = {}
        wrong_coords = []
        # every word that has a conj dependency becomes a key in the conjs dictionary, its values are all words that are
        # connected to the key with a conj dependency
        for dep in sent.dependencies:
            if dep[1] == 'conj' \
                    and dep[0].upos != 'PUNCT' and dep[2].upos != 'PUNCT'\
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
            temp_coord = [] # this will be used if what the loop is looking at could be a nested coordination
            cc = None # the conjunction of a coordination
            nested_coord = False
            conjs[l].sort()
            for conj in conjs[l]:
                # an element of coordination is added either to the main coordination or to the temporary (possibly nested) one
                if cc:
                    temp_coord.append(conj)
                else:
                    crd.append(conj)

                # this loop looks for a conjunction attached to the element of a coordination
                for ch in sent.words[conj-1].children:
                    if not cc and sent.words[ch-1].deprel == 'cc':
                        # sets a conjunction if there wasn't any found yet
                        cc = sent.words[ch-1].text
                    elif cc and sent.words[ch-1].deprel == 'cc' and sent.words[ch-1].text != cc:
                        # if there was a conjunction found already and now there's a different one, we're looking at
                        # a nested coordination
                        crds.append(crd)
                        crd = [l] + temp_coord
                        nested_coord = True
                        cc = sent.words[ch-1].text
                    elif cc and sent.words[ch-1].deprel == 'cc' and sent.words[ch-1].text == cc:
                        # if there was a conjunction found already and the new one is the same, we can clean
                        # the temp_coord list, because so far this seems to not be a nested coordination
                        crd += temp_coord
                        temp_coord = []
            if not temp_coord:
                crds.append(crd)

        if crds:
            # if there are any valid conj dependencies in a sentence, it will be included in the .conllu file
            # a sentence id including the @@ marker from COCA source files is assigned and will be both in the .conllu
            # and .csv file
            sent.coca_sent_id = str(marker) + '-' + str(index)
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
                coord['sent_id'] = sent.coca_sent_id
                coordinations.append(coord)

    sentence_count += len(doc.sentences)
    return coordinations, sentence_count


def extract_coords_from_conll(doc):
    coordinations = []
    for sent in tqdm(doc.sentences):
        # updating sentence count so that it corresponds to the sentence ids in the source .tsv file
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
            nested_coord = False
            for conj in conjs[l]:
                # an element of coordination is added either to the main coordination or a temporary (possibly nested) one
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
                        nested_coord = True
                        cc = sent.words[ch - 1].text
                    elif cc and sent.words[ch - 1].deprel == 'cc' and sent.words[ch - 1].text == cc:
                        # if there was a conjunction found already and the new one is the same, we can clean
                        # the temp_coord list, because so far this seems to not be a nested coordination
                        crd += temp_coord
                        temp_coord = []
            if not temp_coord:
                crds.append(crd)

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
                    if sent.words[child - 1].deprel == 'cc':
                        coord['conj'] = sent.words[child - 1]
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
                sent_conll.append('# sent_id = ' + sentence.coca_sent_id)
            else:
                sent_conll.append(com)
        for token in sentence.tokens:
            sent_conll.extend(token.to_conll_text().split('\n'))
        doc_conll.append(sent_conll)

    path = os.getcwd() + '/done-conll/stanza_trees_' + str(genre) + '_' + str(year) + '.conllu'
    with open(path, mode='w', encoding='utf-8') as conll_file:
        for sent in doc_conll:
            for line in sent:
                conll_file.write(line + '\n')
            conll_file.write('\n')


def create_csv(crd_list, genre, year, source):
    """
    creates a .csv table where every row has information about one coordination
    :param crd_list: list of dictionaries, each dictionary represents a coordination
    :param genre: genre of the parsed file, information included in the filename and in the table
    :param year: year in which the parsed text was published, included in the filename and the table
    :param source: name of the source file
    :return: nothing
    """
    path = os.getcwd() + '/done-csv/stanza_coordinations_' + str(genre) + '_' + str(year) + '.csv'
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
                             source])                                                                                   # converted.from.file


# running --------------------------------------------------------------------------------------------------------------
def run(path):
    s = datetime.now()

    txts, genre, year, source = chunker(path)
    crds_full_list = []
    conll_list = []
    sent_count = 0

    # extracts coordinations one chunk at a time
    for mrk in tqdm(txts.keys()):
        coordinations, sent_count = extract_coords(txts[mrk], mrk, conll_list, sent_count)
        crds_full_list += coordinations

    print('creating a csv...')
    create_csv(crds_full_list, genre, year, source)
    print('csv created!')

    print('processing conll...')
    create_conllu(conll_list, genre, year)
    print('done!')

    e = datetime.now()
    print(e-s)


def run_from_conll(file):
    s = datetime.now()
    doc = CoNLL.conll2doc(os.getcwd() + '/done-conll/' + file)
    e = datetime.now()
    print(e-s)
    coordinations = extract_coords_from_conll(doc)
    genre = re.search('acad|news|fic|mag|blog|web|tvm', file).group()
    year = re.search('[0-9]+', file).group()
    print('writing to csv...')
    create_csv(coordinations, genre=genre, year=year, source=f'text_{genre}_{year}.txt')
    print('done!\n' + 30*'--')


# normalna wersja do terminala:

# for file in args.f:
#     print('processing ' + file)
#     with torch.no_grad():
#         run(os.getcwd() + '/split/' + file)

# jakieś moje nienormalne wersje nwm:

run_from_conll('acad_2048.conllu')

# genre = args.f[0]
# if args.f[1] == '7':
#     files = [f'split_{genre}_{year}.tsv' for year in range(1991, 2007, 2)]
#     files.remove(f'split_{genre}_2001.tsv')
# elif args.f[1] == '6':
#     files = [f'split_{genre}_{year}.tsv' for year in range(2007, 2021, 2)]
#     files.remove(f'split_{genre}_2011.tsv')
# for file in files:
#     print('processing ' + file)
#     with torch.no_grad():
#         run(os.getcwd() + '/split/' + file)
# PAMIĘTAJ ŻEBY WYPAKOWAĆ PLIKI ZANIM TO PUŚCISZ

# files = [f'stanza_trees_acad_{year}.conllu' for year in range(2008, 2020, 2)]
# print(files)
# for file in files:
#     print('processing ' + file)
#     run_from_conll(file)
