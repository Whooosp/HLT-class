import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys
import random

""""
preprocess() method: takes in a raw_text string and performs preprocessing on it in order to return:
    1. The tokens from the raw text
    2. A dictionary mapping each unique noun from the text to the number of times they appear *as a noun* within
        the text

!!IMPORTANT

    I altered the instructions because it didn't make any sense to do pos tagging to the set of unique lemmas
if the goal was to retrieve all the words in the raw text that were *used* as nouns. Some words can be used
in multiple parts of speech depending on the context, and using the set of unique lemmas will assign a random
ordering of all the lemmas when running the nltk.pos_tag() function, and therefore will not faithfully represent
the words that were used as nouns within the raw text itself. This can be seen manifesting in random results as
for the "most commonly used nouns" since depending on how python decides to order the set object when passing
to the pos_tag() function, then each word could have a different context attached to it each time.

    Instead, I will create a list of lemmas that will retain the order of the tokens as they were in the raw text,
perform pos tagging on that list, and then create a set of tagged lemmas. This ensures consistent results, and also
words that (as far as the nltk package can tell) were definitely used as nouns within the raw text.

    As discussed above, I noticed that the instructions did not account for the ordering of the words when
performing part of speech tagging. Since the preprocess function (according to the instructions) returns words that 
were used as nouns * at least once *, it would only be fair to make sure not to overcount the number of times each 
word was used ** as a noun ** (instead of counting the number of times the word was used in a different pos). 
Therefore I also create the {noun: count} dictionary here and return it since this function already has the list of
tagged lemmas (before it gets turned into a set of unique tagged lemmas) and can easily find the count for each
word ** that's used as a noun ** and ** how many times it was used as a noun **.

"""


def preprocess(raw_text: str):
    # Finding lexical diversity
    tokens = nltk.word_tokenize(raw_text)
    lex_diversity = len(set(tokens)) / len(tokens)
    print(f'Lexical diversity: {lex_diversity:.2f}')

    # Tokenizing lower-case raw text
    tokens = [t.lower() for t in tokens
              if t.isalpha() and
              t not in stopwords.words('english') and
              len(t) > 5]

    # Lemmatizing
    lemmatizer = WordNetLemmatizer()

    lemmas = [lemmatizer.lemmatize(t) for t in tokens]

    # POS tagging lemmas
    tagged = nltk.pos_tag(lemmas)

    # Creating the dictionary of unique nouns and their counts
    noun_count = {lemma: tagged.count((lemma, pos)) for lemma, pos in set(tagged) if pos[:2] == 'NN'}

    # Creating a set of tagged lemmas and printing 20
    for tag in list(set(tagged))[:20]:
        print(tag)

    print(f'There are {len(tokens)} tokens and {len(noun_count)} nouns within the text')

    return tokens, noun_count


""" 
Guessing game function 

    Takes a list of nouns as input and creates a guessing game using those nouns. Player starts with 5 points and gains
1 point for each correctly guessed letter and loses 1 point for each incorrectly guessed letter. Once the user has
found the entire word, a new word is given and the old one is thrown out. The game is ended either when the player
has a negative score, enters '!' as input, or when they've guessed all words within the nouns list.
"""


def guessing_game(nouns):
    words_guessed = 0
    points = 5
    word = random.choice(nouns)
    guessed = set()
    while True:
        # Printing word with blank spaces for unguessed letters
        print(''.join(c if c in guessed else '_' for c in word))

        # Entire word has been found
        if set(word) <= guessed:

            # All words have been found
            if len(nouns) == 1:
                print("""Wait, you actually finished all 50 of them... you have way too much free ti-
                       I meeeeaaaaan CONGRATULATIONS!!! Thank you for sinking an inhuman amount of time into this
                       game, possible missing out on important life activities just to play a pretentious version
                       of hangman, really brings a tear to my robot eye. But yeah, you guessed all of them, so this
                       is it. I would say 'play again sometime' but you really shouldn't. This game gets its words
                       from the same text and the ones you guessed were the 50 most common nouns, extracted using NLTK
                       so you'll always get the same 50 words. Tough luck, but hey, there's more to life than a 
                       guessing game, so go try something else out.""")
                print(f'Oh, also you had a whole {points} number of points to spare after guessing all of them too,'
                      f' pretty impressive.')
                exit()

            # Moving on to the next word
            print("You solved it! Time for another word")
            words_guessed += 1
            nouns.remove(word)
            guessed = set()
            word = random.choice(nouns)
            print(''.join(c if c in guessed else '_' for c in word))

        # Guessing a letter
        letter = input("Guess a letter: ")

        # Non-regular input loop
        while not letter.isalpha() or not 0 < len(letter) < 2 or letter in guessed:
            prompt = ""

            # Exit condition
            if letter == '!':
                print("Aww ok, play again sometime! *Exiting game*")
                print(f'You guessed a total of {words_guessed} words with {points} points to spare!')
                return

            # Entered more than 1 letter
            if len(letter) > 1:
                prompt = "A letter is only one letter, not multiple. Try again: "
            # Entered nothing
            elif len(letter) == 0:
                prompt = "Actually typing something would be pretty neat... Try again: "
            # Entered non-alpha character
            elif not letter.isalpha():
                prompt = "W3 b0th kno# th@t wa5n't a |ette&, +ry aga!n: "
            # Letter already guessed
            else:
                prompt = "Whooosp! You've already guessed that letter, try again: "

            # Prompting for a valid letter
            letter = input(prompt)

        # Keeping track of guessed letters
        guessed.add(letter)

        # Letter in word
        if letter in set(word):
            print("Right!", end=' ')
            points += 1

        # Letter not in word
        else:
            # Negative points
            if points == 0:
                print("Sorry, you've lost the game.")
                print(f'You guessed {words_guessed} words, great job!')
                return
            print("Sorry, guess again.", end=" ")
            points -= 1

        print(f'Score is {points}')


def main(file_name):
    raw_text = ""
    with open(file_name) as f:
        # reading in file
        raw_text = f.read()

    # Preprocessing
    tokens, noun_count = preprocess(raw_text)

    # Gathering the 50 most common nouns and printing from least -> most common
    most_common_nouns = []
    for noun, count in sorted(noun_count.items(), key=lambda x: (x[1], x[0]))[-50:]:
        print(f'Noun: {noun} Count: {count}')
        most_common_nouns.append(noun)

    guessing_game(most_common_nouns)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("ERROR: no argument passed for text file")
        exit()
    main(sys.argv[1])
