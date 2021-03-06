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
    character_traits_chatbot = {
        "friendly": ["friendly", "kind", "kindly", "not rude", "n't rude", "not brutal", "n't brutal", "not impolite",
                     "n't impolite",
                     "not bold", "n't bold", "not discourteous", "n't discourteous", "not unmannerly", "n't unmannerly",
                     "not uncivil",
                     "n't uncivil"],
        "happy": ["happy", "glad", "joyous", "not aggressive", "n't aggressive", "not angry", "n't angry", "not mad",
                  "n't mad",
                  "not evil", "n't evil"],
        "aggressive": ["aggressive", "angry", "mad", "evil", "not happy", "n't happy", "not glad", "n't glad",
                       "not joyous", "n't joyous"],
        "rude": ["rude", "brutal", "impolite", "bold", "discourteous", "unmannerly", "uncivil", "not friendly",
                 "n't friendly",
                 "not kind", "n't kind", "not kindly", "n't kindly"],
        "lazy": ["lazy", "idle"],
        "pushy": ["pushy", "pushful", "obtrusive"]}
    character_traits = []

    for entity in entities:
        for ct, cts in character_traits_chatbot.items():
            for c in cts:
                if entity.lower() == c and ct.lower() not in character_traits:
                    character_traits.append(ct)

    if len(character_traits) == 0:
        num = random.randint(0, len(character_traits_chatbot) - 1)
        character_traits.append(list(character_traits_chatbot)[num])

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
    gender_female_keywords = {"she", "female", "her", "woman", "women", "wife", "wives", "girl"}
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
    glasses_keywords = {"wears glasses", "wear glasses", "wearing glasses", "has glasses"}
    no_glasses_keywords = {"wears no glasses", "wear no glasses", "wearing no glasses", "not wear glasses",
                           "no glasses"}
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


def ethnicity_picker(ethnicity_dict):
    ety_list = []
    largest_count = max(ethnicity_dict.values())
    print(ethnicity_dict)
    for ety, counter in ethnicity_dict.items():
        if counter == largest_count:
            ety_list.append(ety)

    print(ety_list)
    num = random.randint(0, len(ety_list) - 1)
    return list(ety_list)[num]


def get_ethnicity(entities):
    ethnicity_keywords = {"caucasian": 0, "african": 0, "southern": 0, "asian": 0}
    for entity in entities:
        for ety, counter in ethnicity_keywords.items():
            if entity.lower() == ety.lower():
                ethnicity_keywords[ety] = counter + 1

    return ethnicity_picker(ethnicity_keywords)


def get_random_name(gender, ethnicity):
    female_names = {
        "caucasian": ["Molly", "Claire", "Abigail", "Jenna", "Allison", "Hannah", "Kaitlin", "Katy", "Emily",
                      "Katherine"],
        "african": ["Nombeko", "Ekua", "Emem", "Anaya", "Ashanti", "Chike", "Mesi", "Nia", "Sauda", "Zalika"],
        "southern": ["Caroline", "Charlotte", "Ruby", "Bea", "Daisy", "Isabelle", "Selena", "Rita", "Ella", "Violet"],
        "asian": ["Kim", "Minji", "Jane", "Lily", "Alice", "Amy", "Jessica", "Sarah", "Rachel", "Cherry"]}

    male_names = {"caucasian": ["Jake", "Cody", "Luke", "Logan", "Cole", "Lucas", "Bradley", "Jacob", "Dylan", "Colin"],
                  "african": ["Akachi", "Berko", "Cayman", "Chibuzo", "Desta", "Dubaku", "Keyon", "Obasi", "Simba",
                              "Talib"],
                  "southern": ["Billy", "Abott", "Alden", "Mason", "Davis", "Nolan", "Redmond", "Victor", "Lester",
                               "Emmet"],
                  "asian": ["Lee", "Jason", "Daniel", "James", "David", "Jack", "Eric", "Tony", "Sam", "Chris"]}

    if gender.lower() == "male":
        for ety, names in male_names.items():
            if ety == ethnicity:
                rand_num = random.randint(0, len(names) - 1)
                return names[rand_num]
    else:
        for ety, names in female_names.items():
            if ety == ethnicity:
                rand_num = random.randint(0, len(names) - 1)
                return names[rand_num]


def get_json(doc):
    character_traits = get_character_traits(filter_list(doc, "ct"))
    age = get_age(filter_list(doc, "age"))
    gender = get_gender(filter_list(doc, "gender"))
    glasses = get_glasses(filter_list(doc, "glasses"))
    ethnicity = get_ethnicity(filter_list(doc, "ety"))
    name = get_random_name(gender, ethnicity)

    json_data = {"Id": 1,
                 "character_traits": character_traits,
                 "age": age,
                 "gender": gender,
                 "glasses": glasses,
                 "ethnicity": ethnicity,
                 "name": name}

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
    "sHe is 28 years old and has a dog. Sometimes she is very rude and aggressive to people. She is a southern.")

# K. Jaiswal. Custom Named Entity Recognition Using Spacy. Geraadpleegd via
# https://towardsdatascience.com/custom-named-entity-recognition-using-spacy-7140ebbb3718
# Geraadpleegd op 4 april 2020

# M. Murugavel. How to Train NER with Custom training data using spaCy. Geraadpleegd via
# https://medium.com/@manivannan_data/how-to-train-ner-with-custom-training-data-using-spacy-188e0e508c6
# Geraadpleegd op 4 april 2020

# M. Murugavel. Train Spacy ner with custom dataset. Geraadpleegd via
# https://github.com/ManivannanMurugavel/spacy-ner-annotator
# Geraadpleegd op 4 april 2020
