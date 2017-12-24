import xmltodict
import spacy
import pandas as pd

def iterate_sentence(path, nlp):
    
    with open(path) as f:
        training_dict = xmltodict.parse(f.read(), strip_whitespace=False)

    for sentence in training_dict["sentences"]["sentence"]:
        sentence_id = sentence["@id"]
        sentence_text = sentence["text"]

        aspect_info = []
        all_terms = sentence.get("aspectTerms", {}).get("aspectTerm", [])
        if type(all_terms) is not list:
            all_terms = [all_terms]
        for aspect_term in all_terms:
            term = aspect_term["@term"]
            idx_from = int(aspect_term["@from"])
            idx_to = int(aspect_term["@to"])
            aspect_info.append((idx_from, idx_to, term))

        sentence_tokens = list(nlp(sentence_text))
        labels = get_bio(sentence_tokens, aspect_info)
        
        yield sentence_tokens, sentence_id, sentence_text, aspect_info, labels
    

def load_data(path, nlp):
    
    training_raw_df = {
        "id": [],
        "text": [],
        "all_aspects": [],
        "token": [],
        "label": []
    }
    
    for sentence_tokens, sentence_id, sentence_text, aspect_info, labels in iterate_sentence(path, nlp):
        for n, t in enumerate(sentence_tokens):
            training_raw_df["id"].append(sentence_id)
            training_raw_df["text"].append(sentence_text)
            training_raw_df["all_aspects"].append([a[-1] for a in aspect_info])
            training_raw_df["token"].append(t)
            training_raw_df["label"].append(labels[n])
 
    return pd.DataFrame(training_raw_df)[["id", "token", "label", "all_aspects", "text"]]

def get_bio(tokens, aspect_term_info, verbose=False):
    if len(aspect_term_info) == 0:
        aspect_term_info = [(float("inf"), -1, "None")]
    aspect_term_info = sorted(aspect_term_info)
    labels = []
    aspect_from, aspect_to, aspect_term = aspect_term_info.pop(0)
    
    for t in tokens:
        t_from = t.idx
        t_to = t_from + len(t.text)
        
        if t_from == aspect_from:
            curr_label = "B"
        elif t_from > aspect_from and t_to <= aspect_to:
            curr_label = "I"
        else:
            curr_label = "O"
        labels.append(curr_label)
        
        if verbose:
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(t_from, t_to, aspect_from, aspect_to, t.text, curr_label, aspect_term))
        
        # complete the current aspect term, loading the next one
        if t_to >= aspect_to:
            if len(aspect_term_info) == 0:
                aspect_from, aspect_to, aspect_term = (float("inf"), -1, "None")
            else:
                aspect_from, aspect_to, aspect_term = aspect_term_info.pop(0)
    if len(aspect_term_info) != 0:
        raise ValueError("Missing some aspect terms \n\t{} \n\n\t{} \n\n\t".format(aspect_term_info, tokens, labels))
    return labels