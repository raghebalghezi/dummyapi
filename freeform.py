from flu import generate_fluency_features
from gram import generate_grammatical_features
from lexical import generate_lexical_features
from asr_features import generate_asr_features
from ta import sim

def DNN_loader(LANG):
    if LANG == 'fi':
        lexgram_eval = ""
        flu_eval = ""
        pron_eval = ""
        hol_eval = ""
    if LANG == 'sv':
        lexgram_eval = ""
        flu_eval = ""
        pron_eval = ""
        hol_eval = ""
    return lexgram_eval, flu_eval, pron_eval, hol_eval



def generate_freeform(audio_path, model,  processor, parser, LANG, prompt):
    asr_res = generate_asr_features(audio_path,  model, processor)
    flu_res = generate_fluency_features(audio_path)
    lex_res = generate_lexical_features(asr_res['transcript'], LANG)
    gram_res = generate_grammatical_features(asr_res['transcript'], parser, LANG)

    lexgram_eval, flu_eval, pron_eval, hol_eval = DNN_loader(LANG)

    lexgram_score = lexgram_eval.predict(lex_res)
    flu_score = flu_eval.predict(flu_res)
    pron_score = pron_eval.predict()
    hol_score = hol_eval.predic(asr_res['embedding'])

    return {"transcript": asr_res['transcript'],
            "task_completion": sim(asr_res['transcript'], prompt), 
            "holistic": hol_score,
            "fluency": {"score": flu_score, 
                        "feature1": "", 
                        "feature2": "", 
                        "feature3": ""},
            "pronunciation": {"score": pron_score, 
                        "feature1": "", 
                        "feature2": "", 
                        "feature3": ""},
            "lexicogrammatical": {"score": lexgram_score, 
                        "feature1": "", 
                        "feature2": "", 
                        "feature3": "",
                        "feature4": "", 
                        "feature5": "", 
                        "feature6": ""}}


