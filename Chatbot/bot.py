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
        score = max(jaro_winkler_similarity(name.lower(), given_name.lower()) if name else 0. for name in names)
        if score >= max_score:
            max_score = score
            max_names = names
    return max_names, max_score


class ChessBot:
    def __init__(self):
        self.clarify = ''
        self.clarified_sentence = ''
        self.entities = None
        self.intent = ''
        self.sub_intent = ''

        self.ner_classifier = AutoModelForTokenClassification.from_pretrained(
            "malduwais/distilbert-base-uncased-finetuned-ner")
        self.ner_tokenizer = DistilBertTokenizerFast.from_pretrained("malduwais/distilbert-base-uncased-finetuned-ner")

        self.ner_to_eng = {'LABEL_1': 'P_Name',
                           'LABEL_2': 'P_Name',
                           'LABEL_3': 'P_T_Name',
                           'LABEL_4': 'P_T_Name',
                           'LABEL_5': 'Location',
                           'LABEL_6': 'Location',
                           'LABEL_7': 'P_T_Name',
                           'LABEL_8': 'P_T_Name'}
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
        self.tourn_df['Year'] = self.tourn_df['Year'].astype(str)

        # print(self.tourn_df[self.tourn_df['Tournament_Location'].str.contains('Reggio')])
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
            if label not in entity_dict:
                entity_dict[label] = [entity]
            else:
                entity_dict[label].append(entity)
        if len(dates) > 0:
            database_entities['Date'] = str(next(d[0].year for d in dates))
        # print(entity_dict)

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
                max_location, max_location_score = None, None
                if intent == 'Tournament':
                    max_location, max_location_score = find_best_match(['Tournament_Location'], tourn_df, location)
                elif intent == 'Person':
                    max_location, max_location_score = max(find_best_match(['Federation'], person_df, location),
                                                           find_best_match(['Birthplace'], person_df, location),
                                                           key=lambda x: x[1])
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
                # print(name)
                # print(entity_dict['P_Name'])
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
                    self.clarified_sentence = sentence.lower().replace(given_name, max_full_name.Full_name)
                    return max_full_name.Full_name, given_name

        return database_entities

    def build_person_msg(self, person_ent, built_msg=''):
        # print(person_ent)
        person_name = person_ent.Full_name
        years_old = int(person_ent.Died[:4]) - int(person_ent.Born[:4]) if person_ent.Died else \
            2022 - int(person_ent.Born[:4])
        location = person_ent.Federation
        title_year = person_ent.TitleYear
        if self.sub_intent == 'Who_is':
            built_msg += f'{person_name} is a {years_old} year old grandmaster from {location}. ' \
                         f'They have held the grandmaster title since {title_year}' if not person_ent.Died else \
                f'{person_name} was a {years_old} year old grandmaster from {location}. ' \
                f'They held the grandmaster title since {title_year} until they passed away in' \
                f'{person_ent.Died}'
        elif self.sub_intent == 'When_is':
            built_msg += f"{person_name} was born on {person_ent.Born}"
            built_msg += f" and died on {person_ent.Died}" if person_ent.Died else ''
        elif self.sub_intent == 'Where_is':
            built_msg += f"{person_name} is from {location}"
        return built_msg

    def build_tourn_msg(self, general_df, won_tourn=False):
        tournament_name = general_df.Tournament_Name.iloc[0]
        tournament_location = general_df.Tournament_Location.iloc[0]
        if len(general_df['Tournament_Name']) == 1:
            winner_name = general_df.winner.iloc[0]
            tournament_year = general_df.Year.iloc[0]
            return f'{winner_name} won the {tournament_year} {tournament_name} at {tournament_location}'
        if not won_tourn:
            self.clarify = "Tourn_year"
            self.entities = general_df, 'Person' in self.entities
            return f"The {tournament_name} has run from {min(general_df['Year'])} to {max(general_df['Year'])} in " \
                   f"{tournament_location}. Please specify a year to know who won that particular tournament."
        if len(general_df['Tournament_Name']) <= 5:
            year_list = sorted(general_df['Year'].tolist())
            year_str = ', '.join(year_list[:-1]) + f"and {year_list[-1]}"
            return f"{self.entities['Person'].Full_name} has won the {self.entities['Tournament']} in {year_str}"
        return f"{self.entities['Person'].Full_name} has won the {self.entities['Tournament']} {len(general_df)}" \
               f" times from {min(general_df['Year'])} to {max(general_df['Year'])}"

    def respond(self, sentence: str):
        # Bot is in error state(s)
        msg = ''
        if self.clarify == 'Person_wrong':
            if len(sentence.split()) > 1:
                return "Sorry, please answer yes or no"
            if sentence.lower().startswith('y') and sentence.lower()[-1] in 'yaesph':
                # print('AFFIRMATIVE ANSWER')
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
                           f'the grandmaster. If you want to move on, though, feel free to just ask a new question!'
                sentence = self.clarified_sentence.replace('%PERSON_NAME%', self.entities[idx])
                self.clarify = ''
                self.clarified_sentence = ''
            else:
                score, gm = max((jaro_winkler_similarity(sentence, ent), ent) if ent else (0., '')
                                for ent in self.entities)
                if score >= .8:
                    sentence = self.clarified_sentence.replace('%PERSON_NAME%', gm)
                    self.clarify = ''
                    self.clarified_sentence = ''
                else:
                    self.clarify = 'skip'
                    self.clarified_sentence = sentence
                    return 'Alright, moving on'
        elif self.clarify == 'Person_mult_valid':
            person_ent = None
            if len(sentence) == 1 and sentence.isnumeric():
                idx = int(sentence) - 1
                if idx < 0 or idx >= len(self.entities):
                    return f'Please either choose a number between 1 and {len(self.entities)} or type the name of' \
                           f'the grandmaster. If you want to move on, though, feel free to just ask a new question!'
                self.clarify = ''
                person_ent = self.entities[idx]
            else:
                score, pot_person_ent = max((jaro_winkler_similarity(sentence, ent.Full_name), ent) if ent else (0., '')
                                            for ent in self.entities)
                if score >= .8:
                    self.clarify = ''
                    person_ent = pot_person_ent
                else:
                    self.clarify = 'skip'
                    self.clarified_sentence = sentence
                    return 'Alright, moving on'
            return self.build_person_msg(person_ent)

        elif self.clarify == 'Tourn_mult':
            tourn_list = self.entities[1]
            general_df = self.entities[0]
            winner = self.entities[2]
            if len(sentence) == 1 and sentence.isnumeric():
                idx = int(sentence) - 1
                if idx < 0 or idx >= len(self.entities):
                    return f'Please either choose a number between 1 and {len(self.entities)} or type the name of' \
                           f'a tournament. If you want to move on, though, feel free to just ask a new question!'
                self.clarify = ''
                general_df = general_df[general_df['Tournament_Name'] == tourn_list[idx]]
                return self.build_tourn_msg(general_df, winner and len(general_df['winner'].unique()) == 1)
            else:
                score, pot_tournament = max((jaro_winkler_similarity(sentence, ent), ent) if ent else
                                           (0., '') for ent in tourn_list)
                if score >= .8:
                    self.clarify = ''
                    general_df = general_df[general_df['Tournament_Name'] == pot_tournament]
                    return self.build_tourn_msg(general_df,
                                                winner and len(general_df['winner'].unique()) == 1)
                else:
                    self.clarify = 'skip'
                    self.clarified_sentence = sentence
                    return 'Alright, moving on'
        elif self.clarify == 'Tourn_year':
            general_df = self.entities[0]
            winner = self.entities[1]
            years = general_df['Year'].tolist()
            if sentence not in years:
                self.clarify = 'skip'
                self.clarified_sentence = sentence
                return 'Alright, moving on'
            self.clarify = ''
            general_df = general_df[general_df['Year'] == sentence]
            return self.build_tourn_msg(general_df, winner and len(general_df['winner'].unique()) == 1)
        elif self.clarify == 'skip':
            self.clarify = ''
            self.clarified_sentence = ''
        # print(sentence)

        self.intent = 'Person'
        self.sub_intent = 'Who_is'
        self.entities = self.extract_database_entities(sentence, self.person_df, self.tourn_df, self.intent)

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

        # print(self.entities)
        if 'Tournament' in self.entities:
            self.intent = 'Tournament'
        msg = ''
        if self.intent == 'Person':
            location = self.entities['Location'] if 'Location' in self.entities else None
            loc_f = lambda x, y: (bool({x.lower() if x else '', y.lower() if y else ''} &
                                       set(location.lower().replace(',', '').split()))
                                  if location else True)
            loc_mask = self.person_df.apply(
                lambda row: loc_f(row.Federation, row.Birthplace), axis=1)
            date_mask = self.person_df['Born'].apply(
                lambda val: (val.startswith(str(self.entities['Date'])) if 'Date' in self.entities else True))
            person_mask = self.person_df['Full_name'].apply(
                lambda val: (val.lower() == self.entities['Person'].Full_name.lower()
                             if 'Person' in self.entities else True))
            general_df = self.person_df[loc_mask & date_mask & person_mask]
            # print(general_df)
            if len(general_df) > 1:
                msg = f"There are {len(general_df)} grandmasters"
                msg += f" from {self.entities['Location']}" if 'Location' in self.entities else ''
                msg += f" born in {self.entities['Date']}" if 'Date' in self.entities else ''
                msg += f" named {self.entities['Person'].Full_name}" if 'Person' in self.entities else ''
                msg += '. Here are a few that I think might interest you:\n'
                person_list_tuples = list(general_df.sample(n=min(len(general_df), 5)).itertuples(index=False))
                person_list = [f"{i + 1}. {ent.Full_name} ID: {ent._2}"
                               for i, ent in enumerate(person_list_tuples)]
                msg += '\n'.join(person_list)
                msg += '\n Please feel free to either type the number or name of the grandmaster you want to know' \
                       'about, or just ask another question if you would rather move on.'
                self.clarify = 'Person_mult_valid'
                self.entities = person_list_tuples
                return msg
            # Exactly one match
            elif len(general_df) == 1:
                queried_player = next(general_df.itertuples(index=None))
                # Given name matches queried name, and there were queries beyond just name
                if 'Person' in self.entities and len(self.entities) > 1:
                    msg += 'Yes, '
                # There was no given name
                else:
                    self.entities['Person'] = queried_player
            # Given name doesn't match query parameters
            elif 'Person' in self.entities:
                msg += 'No, '
            # Query parameters return nothing, no database entity to retrieve
            else:
                msg = 'Sorry, there is no player'
                msg += f" from {self.entities['Location']}" if 'Location' in self.entities else ''
                msg += f" born in {self.entities['Date']}" if 'Date' in self.entities else ''
                return msg + '.'

            # Returning information about person
            if 'Person' in self.entities:
                person_ent = self.entities['Person']
                return self.build_person_msg(person_ent, msg)

        elif self.intent == 'Tournament':
            location = self.entities['Location'] if 'Location' in self.entities else None
            loc_mask = self.tourn_df['Tournament_Location'].apply(
                lambda x: (x.lower() in set(location.lower().replace(',', '').split()) if location else True))
            date_mask = self.tourn_df['Year'].apply(
                lambda val: (val.startswith(self.entities['Date'])
                             if 'Date' in self.entities else True))

            general_df = self.tourn_df[loc_mask & date_mask]

            won_tourn = False
            if len(general_df) == 0:
                msg = f"I couldn't think of any tournaments"
                msg += f" in {location}" if location else ''
                msg += f" during {self.entities['Date']}" if 'Date' in self.entities else ''
                return msg + '.'

            if 'Person' in self.entities and 'Tournament' in self.entities:
                winner_mask = general_df['winner'].apply(
                    lambda val: val.lower() == self.entities['Person'].Full_name.lower() if val else False)
                tourn_mask = general_df['Tournament_Name'].apply(
                    lambda val: (val.lower() == self.entities['Tournament'].lower()
                                 if 'Tournament' in self.entities else True))
                mask = winner_mask & tourn_mask
                if any(mask):
                    msg += 'Yes, '
                    won_tourn = True
                    general_df = general_df[mask]
                else:
                    msg += 'No, '
                    general_df = general_df[tourn_mask]
            elif 'Person' in self.entities:
                winner_mask = general_df['winner'].apply(
                    lambda val: val.lower() == self.entities['Person'].Full_name.lower())
                if any(winner_mask):
                    msg += 'Yes, ' if len(self.entities) > 1 else ''
                    general_df = general_df[winner_mask]
                else:
                    msg += f"{self.entities['Person'].Full_name} has won no prominent tournaments"
                    msg += f" in {location}" if location else ''
                    msg += f" during {self.entities['Date']}" if 'Date' in self.entities else ''
                    return msg
            elif 'Tournament' in self.entities:
                tourn_mask = general_df['Tournament_Name'].apply(
                    lambda val: (val.lower() == self.entities['Tournament'].lower()
                                 if 'Tournament' in self.entities else True))
                msg += 'Yes, ' if len(self.entities) > 1 else ''
                general_df = general_df[tourn_mask]
            # self.entities['Tournament'] = general_df
            # tournament = self.entities['Tournament']
            if len(general_df['Tournament_Name'].unique()) > 1:
                msg += f"I can think of multiple tournaments"
                msg += f" from {location}" if location else ''
                msg += f" held in {self.entities['Date']}" if 'Date' in self.entities else ''
                msg += f" won by {self.entities['Person'].Full_name}" if 'Person' in self.entities else ''
                msg += '. Here are a few that I think you might wanna look into: '
                tourn_list = general_df.sample(n=min(len(general_df), 5))['Tournament_Name'].unique().tolist()
                tourn_list_str = [f'{i + 1}. {ent}'
                                  for i, ent in enumerate(tourn_list)]
                msg += '\n'.join(tourn_list_str)
                msg += '\n Please feel free to either type the number or name of the tournament you want to ' \
                       'know about, or just ask another question if you would rather move on.'
                self.clarify = 'Tourn_mult'
                self.entities = (general_df, tourn_list, 'Person' in self.entities)
                return msg
            return self.build_tourn_msg(general_df, won_tourn)


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
        msg = input().lower().replace('chess', '') if bot.clarify != 'skip' else bot.clarified_sentence
        # for name in bot.person_df['First_name'].unique():
        #     msg = f'Who is {name}'
        if msg == 'logout':
            print('cya')
            break
        print(bot.respond(msg))

    end_time = time.time()
    print(end_time - start_time)
