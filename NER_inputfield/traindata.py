def test():
    # ct = character_trait
    TRAIN_DATA = [
        ('I want a person who is very aggressive and happy sometimes',
         {'entities': [(28, 38, 'ct'), (43, 48, 'ct')]}),

        ('I want a person who is very happy and friendly sometimes',
         {'entities': [(28, 33, 'ct'), (38, 46, 'ct')]}),

        ('I want a person who is very aggressive, rude and happy sometimes',
         {'entities': [(28, 38, 'ct'), (40, 44, 'ct'), (49, 54, 'ct')]}),

        ('I want a person who is happy all the time.',
         {'entities': [(23, 28, 'ct')]}),

        ('I want a person who is friendly all the time.',
         {'entities': [(23, 31, 'ct')]}),

        ('I want a person who is rude all the time.',
         {'entities': [(23, 27, 'ct')]}),

        ('I want a person who is aggressive all the time.',
         {'entities': [(23, 33, 'ct')]}),

        ('I want a person who is pushy all the time.',
         {'entities': [(23, 28, 'ct')]}),

        ('I want a person who is lazy all the time.',
         {'entities': [(23, 27, 'ct')]}),

        ('I want a person who is friendly, rude and lazy sometimes',
         {'entities': [(23, 31, 'ct'), (33, 37, 'ct'), (42, 46, 'ct')]}),

        ('I want a person who is lazy and rude sometimes',
         {'entities': [(23, 27, 'ct'), (32, 36, 'ct')]}),

        ('I want a person who is enormous lazy and sometimes happy',
         {'entities': [(32, 36, 'ct'), (51, 56, 'ct')]}),

        ('I want a person who is not lazy, but happy',
         {'entities': [(23, 31, 'ct'), (37, 42, 'ct')]}),

        ('I want a person who is not happy, but aggressive',
         {'entities': [(23, 32, 'ct'), (38, 48, 'ct')]}),

        ('I want a person who is not aggressive and not happy',
         {'entities': [(23, 37, 'ct'), (42, 51, 'ct')]}),

        ('I want a person who is not rude, but happy',
         {'entities': [(23, 31, 'ct'), (37, 42, 'ct')]}),

        ('I want a person who is not rude and not aggressive, but happy',
         {'entities': [(23, 31, 'ct'), (36, 50, 'ct'), (56, 61, 'ct')]}),

        ]
    return TRAIN_DATA
