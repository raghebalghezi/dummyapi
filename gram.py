
from diaparser.parsers import Parser
import os
import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def parser_loader(LANG):
    if LANG == 'fi':
        parser = Parser.load('fi_tdt.turkunlp')
    if LANG == 'sv':
        parser = Parser.load('sv_talbanken.KB')

    return parser

LANG = "fi"

# parser = parser_loader("fi")

def generate_grammatical_features(transcript, LANG):
    # dataset = parser.predict(transcript, text=LANG)

    # with open("temp/parse.conllu", 'w') as f:
    #     for s in dataset.sentences:
    #         f.write(str(s))

    # os.system("python3 dependency/ling_monitoring.py -p temp/parse.conllu -t 1")
    df = pd.read_csv("output_results/temp_doc.out", sep="\t", on_bad_lines='skip')
    feature_dict = df.to_dict('r')[0]

    del feature_dict['identifier']
    del feature_dict['n_sentences']
    return feature_dict


print(generate_grammatical_features("Haluaisin juoda maitoa.", 'fi'))