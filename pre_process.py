import json
data={}
with open("data.json","r") as file:
    data = json.load(file)

print("loaded data")    


training_data=[]
for example in data["examples"]:
    temp_dict={}
    temp_dict["text"]=example["content"]
    temp_dict["entities"]=[]
    for annotation in example["annotations"]:
        start=annotation["start"]
        end=annotation["end"]
        label=annotation["tag_name"].upper()
        temp_dict["entities"].append((start,end,label))
    training_data.append(temp_dict)
print("conveted to training data")

from spacy.tokens import DocBin
import spacy
from tqdm import tqdm
from spacy.util import filter_spans


def safe_char_span(doc, start, end, label):
    span = doc.char_span(start, end, label=label, alignment_mode="expand")
    if span is None:
        # fallback: search by substring
        entity_text = doc.text[start:end]
        start_pos = doc.text.find(entity_text)
        if start_pos != -1:
            end_pos = start_pos + len(entity_text)
            span = doc.char_span(start_pos, end_pos, label=label, alignment_mode="expand")
    return span



nlp = spacy.blank('en')
# print(tqdm(training_data))
doc_bin = DocBin()
for training_example in tqdm(training_data):
    
    text = training_example['text']
    labels = training_example['entities']
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in labels:
        span = safe_char_span(doc,start,end,label=label)

        if span is None:
            
            
            continue
            
        else:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.ents = filtered_ents
    doc_bin.add(doc)

doc_bin.to_disk("train.spacy")



with open("ner.json","r") as file:
    data = json.load(file)

print("loaded data2")


from spacy.tokens import DocBin
import spacy
from tqdm import tqdm
from spacy.util import filter_spans

nlp = spacy.blank('en')
# print(""abhi :  " ,tqdm(training_data))
doc_bin = DocBin()
for training_example in tqdm(training_data):
    
    text = training_example['text']
    labels = training_example['entities']
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in labels:
        span =safe_char_span(doc,start,end,label=label)

        if span is None:         
            # print("dekho")
            continue

        else:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.ents = filtered_ents
    doc_bin.add(doc)

doc_bin.to_disk("dev.spacy")