import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
import spacy
import torch
from date_extractor import extract_dates
import time
from transformers import AutoModelForTokenClassification, DistilBertTokenizerFast

ner_classifier = AutoModelForTokenClassification.from_pretrained("malduwais/distilbert-base-uncased-finetuned-ner")
ner_tokenizer = DistilBertTokenizerFast.from_pretrained("malduwais/distilbert-base-uncased-finetuned-ner")

ner_to_eng = {'LABEL_1': 'P_Name',
              'LABEL_2': 'P_Name',
              'LABEL_3': 'T_Name',
              'LABEL_4': 'T_Name',
              'LABEL_5': 'Location',
              'LABEL_6': 'Location',
              'LABEL_7': 'Name',
              'LABEL_8': 'Name'}


def extract_entities(sent: str):
    input_ids = ner_tokenizer(sent, padding=True, return_tensors='pt')
    logits = ner_classifier(**input_ids).logits
    pred_ids = logits.argmax(-1)

    preds = [ner_classifier.config.id2label[t.item()] for t in pred_ids[0]][1:-1]
    print(preds)
    tokens = ner_tokenizer.convert_ids_to_tokens(input_ids['input_ids'][0])[1:-1]
    print(tokens)
    separated_entities = [(token, preds[i]) for i, token in enumerate(tokens) if preds[i] != 'LABEL_0']
    joined_entities = []
    for i, entry in enumerate(separated_entities):
        token, label = entry
        if i > 0 and token.startswith('##'):
            joined_entities[-1][0] += token[2:]
        else:
            joined_entities.append([token, label])
    entities = []
    i = 0
    while i < len(joined_entities):
        token, label = joined_entities[i]
        label_no = int(label[-1])
        if label_no % 2 == 1 and i < len(joined_entities)-1:
            entities.append((token + ' ' + joined_entities[i+1][0],
                             ner_to_eng[joined_entities[i+1][1]]))
            i += 2
        else:
            entities.append((token, ner_to_eng[label]))
            i += 1
    print(entities)
    extract_dates(sent, return_precision=True)


if __name__ == "__main__":
    start_time = time.time()
    print('a'.startswith('abc'))
    print(extract_entities("How did thal abergel do in the keres memorial?"))
    end_time = time.time()
    print(end_time-start_time)


