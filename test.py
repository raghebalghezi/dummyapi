from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from diaparser.parsers import Parser
from flu import generate_fluency_features
from gram import generate_grammatical_features
from lexical import generate_lexical_features
from asr_features import generate_asr_features
import time

LANG = "fi"

def model_loader(LANG):
        if LANG == 'fi':
            model = Wav2Vec2ForCTC.from_pretrained("/var/www/html/digi_rsrc/auto_speak_assess/auto_speak_assess/wav2vec2_models/wav2vec2-large-14.2k-fi-digitala_LAQ_09022022/checkpoint-6552")
            processor = Wav2Vec2Processor.from_pretrained("/var/www/html/digi_rsrc/auto_speak_assess/auto_speak_assess/wav2vec2_models/wav2vec2-large-14.2k-fi-digitala_LAQ_09022022/checkpoint-6552")
        if LANG == 'sv':
            model =  Wav2Vec2ForCTC.from_pretrained("/var/www/html/digi_rsrc/auto_speak_assess/auto_speak_assess/wav2vec2_models/wav2vec2_large_voxrex_KBLab_vocab_sv_digitala_LAQ_14012022/checkpoint-8560")
            processor = Wav2Vec2Processor.from_pretrained("/var/www/html/digi_rsrc/auto_speak_assess/auto_speak_assess/wav2vec2_models/wav2vec2_large_voxrex_KBLab_vocab_sv_digitala_LAQ_14012022/checkpoint-8560")
        return model, processor

def parser_loader(LANG):
    if LANG == 'fi':
        parser = Parser.load('fi_tdt.turkunlp')
    if LANG == 'sv':
        parser = Parser.load('sv_talbanken.KB')

    return parser



parser = parser_loader(LANG)
model, processor = model_loader(LANG)



start = time.time()
asr_res = generate_asr_features("8_1_joens_3hv_16.wav",  model, processor)
flu_res = generate_fluency_features("8_1_joens_3hv_16.wav")
lex_res = generate_lexical_features(asr_res['transcript'], LANG)
gram_res = generate_grammatical_features(asr_res['transcript'], parser, LANG)

print(asr_res, flu_res, lex_res, gram_res)
print("--- %s seconds ---" % (time.time() - start))



