def scorer(model_name:str):
    import json
    import spacy
    from spacy.tokens import Doc, Span
    from spacy.training import Example
    from spacy.scorer import Scorer

    with open("data2.json", "r", encoding="utf8") as f:
        test_data = json.load(f)
    LABEL_MAP = {
        "CHARACTER": "PERSON",
        "PERSON": "PERSON",
        "HOUSE": "ORG",
        "ORGANIZATION": "ORG",
        "LOCATION": "LOC",
        "SPORT": "EVENT",
        "BOOK": "WORK_OF_ART",
        "SUBJECT": "WORK_OF_ART",
        "SPELL": "EVENT",
        "MAGICAL_ITEM": "PRODUCT",
        "MAGIC_ITEM": "PRODUCT",
        "CREATURE": "ORG",           # spaCy has no ANIMAL label
        "MAGICAL_CREATURE": "ORG",
    }

    def map_label(label: str):
        return LABEL_MAP.get(label.upper(), label.upper())

    nlp = spacy.load(model_name)

    examples = []

    # input(len(test_data["examples"]))
    for idx,item in enumerate(test_data["examples"]):
        # print(idx)
        text = item["content"]
        doc = nlp.make_doc(text)  
        ents = []

        for ann in item["annotations"]:
            start = ann["start"]
            end = ann["end"]
            if model_name.startswith("en_"):               
                label = map_label(ann["tag_name"])
            else:
                label = ann["tag_name"]
            
                
            span = doc.char_span(start, end, label=label)
            if span:
                ents.append(span)

        gold_doc = Doc(doc.vocab, words=[t.text for t in doc])
        gold_doc.ents = ents

        pred_doc = nlp(text)
        examples.append(Example(pred_doc, gold_doc))
            
    scorer = Scorer()
    scores = scorer.score(examples)
    return scores


