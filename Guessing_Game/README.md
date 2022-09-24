# Guessing game (Portfolio assignment chapter 5)

## Purpose
Learning about NLTK tools to tokenize, lemmatize, and especially
pos tag raw text. Plus a little game to make use of these preprocessing
steps

## Overview
This project is a hangman-esque guessing game that first takes in
a text file as input and then preprocesses it using techniques
we learned in class within NLTK.

First, we tokenize the text and display its Lexical Diversity, and then
proceed to lemmatize all the tokens that are
1. Composed of letters (no special symbols or numbers)
2. Not in the NLTK stopword list
3. Are of length > 5

After which, the lemmatized text is POS-tagged and reduced to a set
of unique lemmas that are nouns, stored as a dictionary with each
unique noun pointing to the number of times it appears as a noun
within the text.

Finally, we retrieve the 50 most commonly used nouns within the text
and create a guessing game using them, where the player begins with
5 points and either gains or loses a point for each correctly/incorrectly
guessed letter. If the player ever has negative points, they lose the game,
alternatively they can enter '!' at any time to exit as well.
