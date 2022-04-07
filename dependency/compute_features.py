# -*- coding: utf-8  -*-
# !/usr/bin/python

import operator
from utils import ratio, get_from_dict, dict_distribution
from linguistic_features import Features


def max_depth(token):
    """
    NP-complete problem! Be careful
    Args:
        token:

    Returns: depth of syntactic tree

    """
    if len(token.children) == 0:
        return 0
    else:
        maximum = 0
        for child in token.children:
            depth = max_depth(child)
            if depth > maximum:
                maximum = depth
        return maximum + 1


def compute_features(sentences, dictionary, type_analysis):
    """
    Compute the features defined in the Class Features()

    :param sentences: ([[Sentence]]) list of sentences
    :param dictionary: //
    :return: computed features (dictionary)
    """

    features_values = {
        'n_sentences': len(sentences),
        'n_tokens': 0
    }

    features = Features()

    for sentence in sentences:
        features_values['n_tokens'] += len(sentence.tokens)

        features.max_sentence_trees_depth.append(max_depth(sentence.root))

        for token in sentence.tokens:  # Features for each token in the sentence
            if dictionary:
                features.lexicon_in_dictionary(token, dictionary)  # Lessico nel dizionario di demauro
            features.count_chars_and_tokens(token)  # Count character per token and number of tokens
            features.count_forms_and_lemmas(token) # Features about forms and lemmas
            features.count_pos_and_dep(token)  # Count uPOS, xPOS, dep
            features.count_lexical_words(token)  # Count lexical words (PAROLE PIENE)
            features.verbal_features(token)  # Verbal features
            features.count_roots(token)
            features.count_links(token)  # Checking number of roots and links per file
            features.count_subjects(token)  # Count preverbal and postverbal subjects
            features.count_objects(token)  # Count preverbal and postverbal objects
            features.count_prepositional_chain_and_syntagms(token, sentence)  # Count prepositional chains and prepositional syntagms
            features.count_subordinate_propositions(token, sentence)  # Count subordinate propositions, pre and post verbal subordinates, subordinate chains

    # Compute type/token ratio on forms and lemmas
    if type_analysis == 1:
        if len(features.ttrs_form) > 0:
            features_values['ttr_form'] = dict_distribution(features.ttrs_form, 'ttr_form')
            features_values['ttr_lemma'] = dict_distribution(features.ttrs_lemma, 'ttr_lemma')
    if type_analysis == 0:
        features_values['ttr_form'] = ratio(len(features.types_form), float(features.n_tok))
        features_values['ttr_lemma'] = ratio(len(features.types_lemma), float(features.n_tok))

    features_values['tokens_per_sent'] = ratio(features_values['n_tokens'], float(features_values['n_sentences']))

    features_values['char_per_tok'] = ratio(features.n_char, float(features.n_tok_no_punct))  # mean chars per token

    if dictionary:
        features_values['in_dict'] = ratio(features.in_dict, float(features.n_tok_no_punct))
        features_values['in_dict_types'] = ratio(features.in_dict_types, float(len(features.types_lemma)))
        features_values['in_FO'] = ratio(features.n_FO, float(features.n_tok_no_punct))
        features_values['in_AD'] = ratio(features.n_AD, float(features.n_tok_no_punct))
        features_values['in_AU'] = ratio(features.n_AU, float(features.n_tok_no_punct))
        features_values['in_FO_types'] = ratio(features.n_FO_types, float(len(features.types_lemma)))
        features_values['in_AD_types'] = ratio(features.n_AD_types, float(len(features.types_lemma)))
        features_values['in_AU_types'] = ratio(features.n_AU_types, float(len(features.types_lemma)))

    features_values['upos_dist'] = dict_distribution(features.upos_total, 'upos_dist')  # Coarse-grained
    features_values['xpos_dist'] = dict_distribution(features.xpos_total, 'xpos_dist')  # Fine-grained
    features_values['lexical_density'] = ratio(features.lexical_words, features.n_tok_no_punct)
    features_values['verbs_mood_dist'] = dict_distribution(features.verbs_mood_total, 'verbs_mood_dist')
    features_values['verbs_tense_dist'] = dict_distribution(features.verbs_tense_total, 'verbs_tense_dist')
    features_values['verbs_gender_dist'] = dict_distribution(features.verbs_gender_total, 'verbs_gender_dist')
    features_values['verbs_form_dist'] = dict_distribution(features.verbs_form_total, 'verbs_form_dist')
    features_values['verbs_num_pers_dist'] = dict_distribution(features.verbs_num_pers_total, 'verbs_num_pers_dist')

    # syntactic features
    features_values['verbal_head_total'] = get_from_dict(features.upos_total, 'VERB')
    features_values['verbal_head_per_sent'] = ratio(get_from_dict(features.upos_total, 'VERB'), features_values['n_sentences'])  # For documents
    features_values['verbal_root_total'] = features.n_verbal_root
    features_values['verbal_root_perc'] = ratio(features.n_verbal_root, features.n_root)  # For documents
    features_values['avg_token_per_clause'] = ratio(features.n_tok, features.n_verb)
    features_values['avg_links_len'] = ratio(features.total_links_len, features.n_links)
    features_values['max_links_len'] = features.max_links_len
    features_values['avg_max_links_len'] = ratio(features.max_links_len, features_values['n_sentences'])
    features_values['avg_max_depth'] = ratio(sum(features.max_sentence_trees_depth), len(features.max_sentence_trees_depth))  # Documents
    features_values['dep_dist'] = dict_distribution(features.dep_total, 'dep_dist')
    features_values['dep_total'] = [('dep_total_' + x, y) for x, y in
                                       sorted(features.dep_total.items(), key=operator.itemgetter(1), reverse=True)]
    features_values['subj_pre'] = ratio(features.n_subj_pre, features.n_subj_pre + features.n_subj_post)
    features_values['subj_post'] = ratio(features.n_subj_post, features.n_subj_pre + features.n_subj_post)
    features_values['obj_pre'] = ratio(features.n_obj_pre, features.n_obj_pre + features.n_obj_post)
    features_values['obj_post'] = ratio(features.n_obj_post, features.n_obj_pre + features.n_obj_post)
    features_values['n_prepositional_chains'] = features.n_prepositional_chain
    features_values['avg_prepositional_chain_len'] = ratio(features.total_prepositional_chain_len, features.n_prepositional_chain)
    features_values['prepositional_chain_total'] = sorted(
        {'prep_total_' + str(i): features.prep_chains.count(i) for i in set(features.prep_chains)}.items(),
        key=operator.itemgetter(1), reverse=True)
    features_values['prepositional_chain_distribution'] = sorted(
        {'prep_dist_' + str(i): features.prep_chains.count(i) / float(features.n_prepositional_chain) for i in
         set(features.prep_chains)}.items(), key=operator.itemgetter(1), reverse=True)
    features_values['subordinate_chains_total'] = sorted(
        {'subordinate_total_' + str(i): features.subordinate_chains.count(i) for i in set(features.subordinate_chains)}.items(),
        key=operator.itemgetter(1), reverse=True)
    features_values['subordinate_chains_distribution'] = sorted(
        {'subordinate_dist_' + str(i): features.subordinate_chains.count(i) / float(features.n_subordinate_chain) for i
         in set(features.subordinate_chains)}.items(), key=operator.itemgetter(1), reverse=True)

    features_values['total_subordinate_proposition'] = features.n_subordinate_proposition
    features_values['total_subordinate_chain'] = features.n_subordinate_chain
    features_values['total_subordinate_chain_len'] = features.total_subordinate_chain_len
    features_values['avg_subordinate_chain_len'] = ratio(features.total_subordinate_chain_len, features.n_subordinate_chain)
    features_values['principal_proposition_dist'] = ratio(features.n_verb - features.n_subordinate_proposition, features.n_verb)
    features_values['subordinate_proposition_dist'] = ratio(features.n_subordinate_proposition, features.n_verb)
    features_values['subordinate_pre'] = ratio(features.n_subordinate_pre, features.n_subordinate_proposition)
    features_values['subordinate_post'] = ratio(features.n_subordinate_post, features.n_subordinate_proposition)
    features_values['verb_edges_dist'] = [('verb_edges_dist_' + str(k), v) for k, v in dict_distribution(features.verb_edges_total, '')]  # Arità totale
    features_values['avg_verb_edges'] = ratio(features.total_verb_edges, features.n_verb)  # Arità media

    return features_values
