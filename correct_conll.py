# corrects the sentence id in the even web22-34 conll files from trankit, because it looked like this:
# SENTENCE : blah blah human speech
# ID : @@2137420
# -25
import os
import re


def clean_sent_id(path):
    with open(path, mode='r', encoding='utf-8') as inp:
        lines = inp.readlines()
    with open(path, mode='w', encoding='utf-8') as outp:
        corrected = []
        last_line = -1
        for n, line in enumerate(lines):
            if last_line == n:
                continue
            if re.match('# ID :', line) and len([x for x in re.finditer('\\t', lines[n+1])]) != 9:
                line = line[:-1] + lines[n+1]
                last_line = n+1
            corrected.append(line)
        outp.writelines(corrected)


path_to_dir = os.getcwd() + '/inp/fic/'
for file in os.listdir(path_to_dir):
    clean_sent_id(path_to_dir + file)
