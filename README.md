# Extracting coordinations from CoNLL-U files or from raw text
The script looks for coordinations in parsed sentences and returns a csv table with information about every coordination that has been found.


## Setup
1. Download zip or clone the repository: `git clone https://github.com/bmagdab/LGPB23-24.git`
2. Install the packages required for the script to run: `pip install -r requirements.txt`


## Usage
Recommended and easiest way to run the script:
1. Put all of the files you want to process in the /inp folder
2. Run the script using `python main.py -d` and a flag indicating which parser produced the conllu files you want to process: -s for stanza, -t for trankit, -c for combo

Therefore if you want to run the script on combo conllu files (to do ciebie Oskar), you will have to run `python main.py -d -c` and everything in the /inp folder will be processed (unless it was already processed and there is a corresponding table in the /outp folder, in that case the file is skipped).

The /inp folder is for input files and for input files only! Do not put anything else in there!!
The csv tables will appear in the /outp folder.

If you want to process specific files instead of the whole input directory, use the flag `-f` and list the files you want to process, for example to process the example file that is in the repository you need to run `python main.py -c -f combo_acad_2137.conllu`.

The script prints out which file is being processed and how much time it takes. Sometimes if the script runs for a while, files start to take longer to be processed. In that case, pressing `CTRL+C` to stop the script and running it again can help (it's best to press the `CTRL+C` soon after reading a conllu file starts, it's quicker then).
