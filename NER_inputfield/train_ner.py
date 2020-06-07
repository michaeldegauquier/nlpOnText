import spacy
import random
from NER_inputfield import traindata

TRAIN_DATA = traindata.test()
trainable = False


def train_spacy(data, iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


try:
    spacy.load('../NER_inputfield/ner_model')
    trainable = True  # Must be False
except IOError:
    trainable = True

if trainable:
    prdnlp = train_spacy(TRAIN_DATA, 20)

    # Save our trained Model
    prdnlp.to_disk('../NER_inputfield/ner_model')

    test_text = ""

    # Test your text
    while test_text != "q":
        test_text = input("Enter your testing text: ")
        doc = prdnlp(test_text)

        entities = ''
        i = 0

        for ent in doc.ents:
            # print(ent.text, ent.start_char, ent.end_char, ent.label_)
            print('{}: ({}, {}, \"{}\")'.format(ent.text, ent.start_char, ent.end_char, ent.label_))
            if len(doc.ents) - 1 == i:
                entities += '({}, {}, \"{}\")'.format(ent.start_char, ent.end_char, ent.label_)
            else:
                entities += '({}, {}, \"{}\"), '.format(ent.start_char, ent.end_char, ent.label_)
                i += 1

        print('("{}",\n{{"entities": [{}]}}),'.format(test_text, entities))
else:
    nlp = spacy.load('../NER_inputfield/ner_model')

    prdnlp = nlp

    test_text = ""

    # Test your text
    while test_text != "q":
        test_text = input("Enter your testing text: ")
        doc = prdnlp(test_text)

        entities = ''
        i = 0

        for ent in doc.ents:
            # print(ent.text, ent.start_char, ent.end_char, ent.label_)
            print('{}: ({}, {}, \"{}\")'.format(ent.text, ent.start_char, ent.end_char, ent.label_))
            if len(doc.ents) - 1 == i:
                entities += '({}, {}, \"{}\")'.format(ent.start_char, ent.end_char, ent.label_)
            else:
                entities += '({}, {}, \"{}\"), '.format(ent.start_char, ent.end_char, ent.label_)
                i += 1

        print('("{}",\n{{"entities": [{}]}}),'.format(test_text, entities))

# K. Jaiswal. Custom Named Entity Recognition Using Spacy. Geraadpleegd via
# https://towardsdatascience.com/custom-named-entity-recognition-using-spacy-7140ebbb3718
# Geraadpleegd op 4 april 2020

# M. Murugavel. How to Train NER with Custom training data using spaCy. Geraadpleegd via
# https://medium.com/@manivannan_data/how-to-train-ner-with-custom-training-data-using-spacy-188e0e508c6
# Geraadpleegd op 4 april 2020

# M. Murugavel. Train Spacy ner with custom dataset. Geraadpleegd via
# https://github.com/ManivannanMurugavel/spacy-ner-annotator
# Geraadpleegd op 4 april 2020
