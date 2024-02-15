# code from Michał Woźniak, email: m.wozniak60@student.uw.edu.pl, with a small adjustment from me in the count_word function

import nltk
from nltk.corpus import cmudict
import re
import inflect
d = cmudict.dict()


my_d = {"$": 2, "%": 2, "&": 1,  "1\\/2": 2, "1\\/4": 3, "3\\/4": 3,
        "jan.": 4, "feb.": 4, "mar.": 1, "apr.": 2, "jun.": 1, "jul.": 2,
        "aug.": 2, "sept.": 3, "oct.": 3, "nov.": 3, "dec.": 3, "'s": 0}


def syllables(word):
    # referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return count


def nsyl(word):
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]
    except KeyError:
        # if word not found in cmudict
        if re.fullmatch(r"[A-Z]{,4}", word):
            n = 0
            for letter in word:
                n += nsyl(letter)[0]
            return [n]
        else:
            return [syllables(word)]


def num_syl(number):
    p = inflect.engine()
    if re.fullmatch(r"19[6-9]\d", number):
        n_syllables = 2 + nsyl(p.number_to_words(number[2]+"0"))[0] + nsyl(p.number_to_words(number[3]))[0]
        if number[3] == "0":
            n_syllables -= 2
        return n_syllables
    
    pat = r"[a-zA-z]+"
    
    s = p.number_to_words(number)
    words = re.findall(pat, s)
    
    n_syllables = 0
    for word in words:
        n_syllables += nsyl(word)[0]
        
    return n_syllables


def count_word(word):
    if word.lower() in my_d:
        return my_d[word.lower()]
    # added this if below to ensure that a word is first looked up in the cmudict - MB
    if word.lower() in d:
        return nsyl(word)[0]

    pat1 = r"\d+s|'\d+s"
    pat2 = r"(?:\d+|\d{1,3}(?:,\d{3})*)(?:\.\d+)?(?:st|nd|rd|th)?"
    
    if re.fullmatch(pat1, word):
        return num_syl(re.findall(r"\d+", word)[0])
    else:
        if re.fullmatch(pat2, word):
            return num_syl(word)
        else:
            n_syllables = 0

            words = re.findall(r"[A-Za-z]+", word)

            nums = re.findall(r"(?:\d+|\d{1,3}(?:,\d{3})*)(?:\.\d+)?", word)

            for w in words:
                n_syllables += nsyl(w)[0]

            for num in nums:
                n_syllables += num_syl(num)
            
    return n_syllables


def count_syl(tree):
    
    pos = tree.treepositions('leaves')
    # print(pos)
    
    n_syllables = 0
    for i, word in enumerate(tree.leaves()):
        lab = tree[pos[i][:-1]].label()
        if lab[0] == "-" and lab[-1] == "-":
            continue
        else:
            n_syllables += count_word(word)
    return n_syllables
