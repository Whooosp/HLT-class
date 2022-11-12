import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.metrics.distance import jaro_winkler_similarity
import spacy
import torch
import pandas as pd
from date_extractor import extract_dates
import time
from transformers import AutoModelForTokenClassification, DistilBertTokenizerFast
import sqlite3
import unicodedata


def find_best_match(keys, df, given_name):
    max_score = 0.8
    max_names = ''
    for names in df[keys].drop_duplicates().itertuples(index=False):
        score = max(jaro_winkler_similarity(name.lower(), given_name.lower()) for name in names)
        if score >= max_score:
            max_score = score
            max_names = names
    return max_names, max_score


class ChessBot:
    def __init__(self):
        self.clarify = ''
        self.clarified_sentence = ''
        self.entities = None

        self.ner_classifier = AutoModelForTokenClassification.from_pretrained(
            "malduwais/distilbert-base-uncased-finetuned-ner")
        self.ner_tokenizer = DistilBertTokenizerFast.from_pretrained("malduwais/distilbert-base-uncased-finetuned-ner")

        self.ner_to_eng = {'LABEL_1': 'P_Name',
                           'LABEL_2': 'P_Name',
                           'LABEL_3': 'P_T_Name',
                           'LABEL_4': 'P_T_Name',
                           'LABEL_5': 'Location',
                           'LABEL_6': 'Location',
                           'LABEL_7': 'Misc',
                           'LABEL_8': 'Misc'}
        self.person_df = pd.read_sql('SELECT * FROM chess_grandmasters', conn)

        self.person_df['First_name'] = self.person_df['First_name'].apply(
            lambda val: ''.join(unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode()))
        self.person_df['Last_name'] = self.person_df['Last_name'].apply(
            lambda val: ''.join(unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode()))
        self.person_df['Full_name'] = self.person_df[['First_name', 'Last_name']].apply(lambda x: ' '.join(x), axis=1)

        self.tourn_df = pd.read_sql('SELECT * FROM '
                                    'tournament_locations tl left outer join tournament_winners tw '
                                    'on tl.Tournament_Name = tw.Tournament_name ',
                                    conn)
        self.tourn_df['winner'] = self.tourn_df['winner'].apply(
            lambda val: ''.join(unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode()) if val else None)
        self.tourn_df['winner'] = self.tourn_df['winner'].apply(
            lambda val: val[:val.find('  ')] if val and '  ' in val else val
        )
        self.tourn_df['winner'] = self.tourn_df['winner'].apply(
            lambda val: val[:val.find('defeated')] if val and 'defeated' in val else val
        )
        self.tourn_df['winner'] = self.tourn_df['winner'].apply(
            lambda val: val[:val.find('drew')] if val and 'drew' in val else val
        )

        print(self.tourn_df[self.tourn_df['Tournament_Location'].str.contains('Reggio')])
        # print(self.tourn_df.head())
        # self.tourn_winner_df = pd.read_sql('SELECT * FROM tournament_winners', conn)

    def extract_entities(self, sent: str):
        input_ids = self.ner_tokenizer(sent, padding=True, return_tensors='pt')
        with torch.no_grad():
            logits = self.ner_classifier(**input_ids).logits
        pred_ids = logits.argmax(-1)

        preds = [self.ner_classifier.config.id2label[t.item()] for t in pred_ids[0]][1:-1]

        tokens = self.ner_tokenizer.convert_ids_to_tokens(input_ids['input_ids'][0])[1:-1]
        # if 'LABEL_7' in preds or 'LABEL_8' in preds:
        # print(preds)
        # print(tokens)
        separated_entities = [(token, preds[i]) for i, token in enumerate(tokens)]
        joined_entities = []
        for i, entry in enumerate(separated_entities):
            token, label = entry
            if i > 0 and token.startswith('##'):
                joined_entities[-1][0] += token[2:]
            elif i > 0 and joined_entities[-1][0][-1] == '-':
                joined_entities[-1][0] += token
            elif i > 0 and label == joined_entities[-1][1]:
                joined_entities[-1][0] += ' ' + token if token.isalpha() else token
                # print(joined_entities[-1][0])
            else:
                joined_entities.append([token, label])
        # print(joined_entities)
        entities = []
        i = 0
        while i < len(joined_entities):
            token, label = joined_entities[i]
            if label == 'LABEL_0':
                i += 1
                continue
            label_no = int(label[-1])
            if label_no % 2 == 1 and i < len(joined_entities) - 1 and joined_entities[i + 1][1] != 'LABEL_0':
                entities.append((token + ' ' + joined_entities[i + 1][0] if token[-1] != '-' else
                                 token + joined_entities[i + 1][0],
                                 self.ner_to_eng[joined_entities[i + 1][1]]))
                i += 2
            else:
                entities.append((token, self.ner_to_eng[label]))
                i += 1
        # print(entities)
        return entities, extract_dates(sent, return_precision=True)

    def extract_database_entities(self, sentence, person_df, tourn_df, intent):
        entities, dates = self.extract_entities(sentence)
        database_entities = {}
        # print(dates)
        entity_dict = {}
        for entity, label in entities:
            if len(entity) == 1:
                entity += '.'
            if label == 'Misc' or label == 'P_T_Name':
                label = 'P_Name' if intent == 'Person' else 'P_T_Name'
            if label not in entity_dict:
                entity_dict[label] = [entity]
            else:
                entity_dict[label].append(entity)
        if len(dates) > 0:
            database_entities['Date'] = next(d[0].year for d in dates)
        print(entity_dict)

        # Distinctly identify tournament and people names from ambiguous tag
        if 'P_T_Name' in entity_dict:
            for name in entity_dict['P_T_Name']:
                if 'P_Name' in entity_dict and 'Tournament' in database_entities:
                    break
                # Compute similarity for people
                # max_full_name_score = 0.8
                # max_full_name = ''
                # for first_name, last_name, full_name in \
                #         person_df[['First_name', 'Last_name', 'Full_name']].itertuples(index=None):
                #     score = max(jaro_winkler_similarity(first_name.lower(), name.lower()),
                #                 jaro_winkler_similarity(last_name.lower(), name.lower()),
                #                 jaro_winkler_similarity(full_name.lower(), name.lower()))
                #     if score >= max_full_name_score:
                #         max_full_name_score = score
                #         max_full_name = full_name
                # max_tourn_name_score = 0.8
                # max_tourn_name = ''
                # for tourn_name, winner in tourn_df[['Tournament_name', 'winner']].itertuples(index=None):
                #     score = jaro_winkler_similarity(tourn_name.lower(), name.lower())
                #     if score >= max_tourn_name_score:
                #         max_tourn_name_score = score
                #         max_tourn_name = tourn_name
                max_full_name, max_full_name_score = find_best_match(['First_name', 'Last_name', 'Full_name'],
                                                                     person_df,
                                                                     name)
                max_tourn_name, max_tourn_name_score = find_best_match(['Tournament_Name'], tourn_df, name)
                # Similar to person
                if max_full_name_score > max_tourn_name_score and max_full_name:
                    if 'P_Name' not in entity_dict:
                        entity_dict['P_Name'] = [max_full_name[-1]]
                    continue
                # Similar to Tournament name
                elif max_tourn_name:
                    if 'Tournament' not in database_entities:
                        database_entities['Tournament'] = max_tourn_name[-1]

        # Disambiguate Locations from tournament names
        if 'Location' in entity_dict:
            for location in entity_dict['Location']:
                # location_idx = sentence.split().index(location)
                if 'Tournament' not in database_entities:
                    # max_tourn_name_score = 0.8
                    # max_tourn_name = ''
                    # for tourn_name in tourn_df['Tournament_name'].unique():
                    #     score = jaro_winkler_similarity(tourn_name.lower(), location.lower())
                    #     if score >= max_tourn_name_score:
                    #         max_tourn_name_score = score
                    #         max_tourn_name = tourn_name
                    max_tourn_name, max_tourn_name_score = find_best_match(['Tournament_Name'], tourn_df, location)
                    if max_tourn_name:
                        database_entities['Tournament'] = max_tourn_name[-1]
                max_location, max_location_score = find_best_match(['Tournament_Location'], tourn_df, location)
                if max_location:
                    database_entities['Location'] = max_location[-1]
                    break

        # Checking for people
        if 'P_Name' in entity_dict:
            name = tuple(entity_dict['P_Name'][0].split())
            # print(name)
            f_mask = person_df.First_name.apply(lambda x: set((n.lower() for n in name)) & set(x.lower().split()))
            l_mask = person_df.Last_name.apply(lambda x: set((n.lower() for n in name)) & set(x.lower().split()))
            full_mask = person_df.Full_name.apply(lambda x: ' '.join(name).lower() == x.lower())
            full_match = person_df[full_mask]
            partial_match = person_df[f_mask | l_mask]
            red_person_df = full_match if len(full_match) > 0 else partial_match

            if len(red_person_df) > 1:
                self.clarify = 'Person_mult'
                self.clarified_sentence = sentence.lower().replace(' '.join(name), '%PERSON_NAME%')
                return red_person_df['Full_name'].tolist()
            elif len(red_person_df) == 1:
                database_entities['Person'] = next(red_person_df.itertuples(index=None))
            # If person name is incorrect
            else:
                print(name)
                self.clarify = 'Person_wrong'

                given_name = ' '.join(name)
                # max_full_name_score = 0.8
                # max_full_name = ''
                # for first_name, last_name, full_name in \
                #         person_df[['First_name', 'Last_name', 'Full_name']].itertuples(index=None):
                #     score = max(jaro_winkler_similarity(first_name.lower(), given_name.lower()),
                #                 jaro_winkler_similarity(last_name.lower(), given_name.lower()),
                #                 jaro_winkler_similarity(full_name.lower(), given_name.lower()))
                #     if score >= max_full_name_score:
                #         max_full_name_score = score
                #         max_full_name = full_name
                max_full_name, max_full_name_score = find_best_match(['First_name', 'Last_name', 'Full_name'],
                                                                     person_df,
                                                                     given_name)
                if max_full_name:
                    # msg += f'I know of {max_full_name} though, '
                    # print(msg)
                    self.clarified_sentence = sentence.lower().replace(given_name, max_full_name)
                    return max_full_name, given_name

        return database_entities

    def respond(self, sentence: str):
        # Bot is in error state(s)
        if self.clarify == 'Person_wrong':
            if len(sentence.split()) > 1:
                return "Sorry, please answer yes or no"
            if sentence.lower().startswith('y') and sentence.lower()[-1] in 'yaesph':
                print('AFFIRMATIVE ANSWER')
                sentence = self.clarified_sentence
                self.clarified_sentence = ''
                self.clarify = ''
            elif sentence.lower().startswith('n') and sentence.lower()[-1] in 'ohayen':
                self.clarified_sentence = ''
                self.clarify = ''
                return 'Sorry, please re-enter your question then.'
            else:
                return 'Sorry, please answer yes or no'
        elif self.clarify == 'Person_mult':
            if len(sentence) == 1 and sentence.isnumeric():
                idx = int(sentence) - 1
                if idx < 0 or idx >= len(self.entities):
                    return f'Please either choose a number between 1 and {len(self.entities)} or type the name of' \
                           f'the grandmaster'
                sentence = self.clarified_sentence.replace('%PERSON_NAME%', self.entities[idx])
                self.clarify = ''
                self.clarified_sentence = ''
            else:
                score, gm = max((jaro_winkler_similarity(sentence, ent), ent) for ent in self.entities)
                if score >= .8:
                    sentence = self.clarified_sentence.replace('%PERSON_NAME%', gm)
                    self.clarify = ''
                    self.clarified_sentence = ''
                else:
                    return f'Please either choose a number between 1 and {len(self.entities)} or type the name of' \
                           f'the grandmaster'
        print(sentence)

        intent = 'Tournament'
        sub_intent = 'How many'
        self.entities = self.extract_database_entities(sentence, self.person_df, self.tourn_df, intent)

        # Recent query caused bot to enter error state
        if self.clarify == 'Person_wrong':
            sugg_name, given_name = self.entities
            msg = f'There doesn\'t seem to be any chess grandmaster with the name {given_name}\n'
            if sugg_name:
                msg += f'I know of {sugg_name} though, is that who you meant?'
            else:
                self.clarify, self.clarified_sentence = '', ''
            return msg
        elif self.clarify == 'Person_mult':
            person_list = self.entities if len(self.entities) < 5 else self.entities[:5]
            person_list = [f'{i + 1}. {ent}' for i, ent in enumerate(person_list)]
            person_msg = '\n'.join(person_list)
            msg = f'There seem to be many grandmasters with that name, please choose one from the following:\n' \
                  f'{person_msg}'
            return msg

        print(self.entities)

        if intent == 'Person':
            # msg = f'{person_name} is a {years_old} grandmaster from {location}.' \
            #       f'They have held the grandmaster title since {title_year}'
            pass
        elif intent == 'Tournament':
            # msg = f'{winner_name} won the {tournament_year} {tournament_name} at {tourn_location}'
            pass

        # elif intent ==
        else:
            pass


if __name__ == "__main__":
    start_time = time.time()
    # print('a'.startswith('abc'))
    conn = sqlite3.connect("chess_schema")
    bot = ChessBot()
    chat = True
    print('Hi, I\'m Chester 1.0, a chatbot here to answer (hopefully) all your chess history questions!')
    while chat:
        msg = input().lower().replace('chess', '')
        # for name in bot.person_df['First_name'].unique():
        #     msg = f'Who is {name}'
        if msg == 'logout':
            print('cya')
            break
        print(bot.respond(msg))

    end_time = time.time()
    print(end_time - start_time)
