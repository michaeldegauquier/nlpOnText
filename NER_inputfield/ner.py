import spacy
import random
from NER_inputfield import traindata


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


def filter_list(doc, entity_name):
    entities = []

    for ent in doc.ents:
        if ent.label_ == entity_name:
            entities.append(ent.text)

    return entities


def get_character_traits(entities):
    character_traits_chatbot = ["friendly", "happy", "aggressive", "rude", "lazy", "pushy"]
    character_traits = []

    for entity in entities:
        for ct in character_traits_chatbot:
            if entity.lower() == ct and entity.lower() not in character_traits:
                character_traits.append(ct)

    if len(character_traits) == 0:
        num = random.randint(0, len(character_traits_chatbot)-1)
        character_traits.append(character_traits_chatbot[num])

    return character_traits


def get_age(entities):
    possible_ages = []
    sum_ages = 0

    for entity in entities:
        if entity.isdigit():
            if 100 >= int(entity) >= 0:
                possible_ages.append(int(entity))

    if len(possible_ages) != 0:
        for age in possible_ages:
            sum_ages += age

        average = round(sum_ages / len(possible_ages))
        return average
    else:
        age = random.randint(18, 100)
        return age


def get_gender(entities):
    gender_male_keywords = {"he", "male", "him", "his", "man", "men", "husband", "husbands"}
    gender_female_keywords = {"she", "female", "her", "woman", "women", "wife", "wives"}
    male_counter = 0
    female_counter = 0

    for entity in entities:
        if entity.lower() in gender_male_keywords:
            male_counter = male_counter + 1
        elif entity.lower() in gender_female_keywords:
            female_counter = female_counter + 1

    if male_counter == female_counter:
        num = random.randint(0, 1)
        if num == 0:
            return "male"
        else:
            return "female"
    else:
        if male_counter > female_counter:
            return "male"
        else:
            return "female"


def get_glasses(entities):
    glasses_keywords = {"wears glasses", "wear glasses", "wearing glasses"}
    no_glasses_keywords = {"wears no glasses", "wear no glasses", "wearing no glasses", "not wear glasses"}
    glasses_counter = 0
    no_glasses_counter = 0

    for entity in entities:
        if entity.lower() in glasses_keywords:
            glasses_counter = glasses_counter + 1
        elif entity.lower() in no_glasses_keywords:
            no_glasses_counter = no_glasses_counter + 1

    if glasses_counter == no_glasses_counter:
        num = random.randint(0, 1)
        if num == 0:
            return True
        else:
            return False
    else:
        if glasses_counter > no_glasses_counter:
            return True
        else:
            return False


def get_ethnicity(entities):
    ethnicity_keywords = ["caucasian", "african", "southern", "asian"]
    ethnicity = ""

    for entity in entities:
        for ety in ethnicity_keywords:
            if entity.lower() == ety:
                ethnicity = entity.lower()

    if ethnicity == "":
        num = random.randint(0, len(ethnicity_keywords)-1)
        return ethnicity_keywords[num]

    return ethnicity


def get_json(doc):
    character_traits = get_character_traits(filter_list(doc, "ct"))
    age = get_age(filter_list(doc, "age"))
    gender = get_gender(filter_list(doc, "gender"))
    glasses = get_glasses(filter_list(doc, "glasses"))
    ethnicity = get_ethnicity(filter_list(doc, "ety"))

    json_data = {"Id": 1,
                 "character_traits": character_traits,
                 "age": age,
                 "gender": gender,
                 "glasses": glasses,
                 "ethnicity": ethnicity}

    print(json_data)

    return character_traits


def get_json_data_from_input(text_input):
    TRAIN_DATA = traindata.test()

    try:
        spacy.load('../NER_inputfield/ner_model')
        trainable = False  # Must be False
    except IOError:
        trainable = True

    if trainable:
        prdnlp = train_spacy(TRAIN_DATA, 20)

        # Save our trained Model
        prdnlp.to_disk('../NER_inputfield/ner_model')
    else:
        nlp = spacy.load('../NER_inputfield/ner_model')

        prdnlp = nlp

    test_text = text_input
    doc = prdnlp(test_text)

    all_entities = []

    for ent in doc.ents:
        all_entities.append(ent.text)
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

    get_json(doc)


get_json_data_from_input(
    "He is 28 years old and has a dog. Sometimes he is very rude and aggressive to people. But most of the time he is friendly. He is an african who is wearing glasses.")
