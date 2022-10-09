import os

import requests
import re
from bs4 import BeautifulSoup
import string
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd


# Web Crawler
#   Authors: Varin Sikand (vss180000) Aditya Guin (asg180005)

def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True


def scrape_url(url: str, to_save: bool):
    r = requests.get(url)

    if r.status_code >= 400:
        return

    data = r.text
    soup = BeautifulSoup(data, 'html.parser')

    data = soup.find_all(text=True)

    titles = [t for t in data if t.parent.name == "title"]
    title = titles[0] if len(titles) > 0 else url[url.index('://') + 3:].replace('/', '_')
    title = ''.join(c for c in title if c.isalnum())
    title = f'url_{title}'
    print(title)

    if to_save:
        with open(f'{title}.txt', 'w', encoding='utf8') as f:
            text = filter(visible, data)
            text = ' '.join(list(text))
            f.write(text)
    return soup


if __name__ == "__main__":

    soup = scrape_url('https://en.wikipedia.org/wiki/Magnus_Carlsen', False)

    urls = set()

    for link in soup.find_all('a'):
        link_str = str(link.get('href'))
        if ('Carlsen' in link_str or 'carlsen' in link_str) and 'chessbomb' not in link_str and '.no/' not in link_str:
            if link_str.startswith('/url?q='):
                link_str = link_str[7:]
            if '&' in link_str:
                i = link_str.find('&')
                link_str = link_str[:i]
            if link_str.startswith('http') and 'google' not in link_str and link_str not in urls:
                print(link_str)
                if scrape_url(link_str, True):
                    urls.add(link_str)

        if len(urls) > 14:
            break

    all_sentences = set()
    for filename in [name for name in os.listdir(os.getcwd()) if name[:3] == "url"]:
        save_string = ''
        with open(filename, 'r', encoding='utf8') as f:
            for line in f:
                temp_line = ''.join(filter(lambda x: x in set(string.printable), ' '.join(line.split())))
                temp_line = temp_line + '.' if temp_line != '' and temp_line[-1] not in '.!?' else temp_line
                save_string += temp_line + ' ' if temp_line.strip() != '' else temp_line
        sentences = sent_tokenize(save_string)
        all_sentences = all_sentences | set(sentences)
        with open("better" + filename, 'w', encoding='utf8') as f:
            f.write(' '.join(sentences))

    token_counts = Counter()
    for filename in [name for name in os.listdir(os.getcwd()) if name.startswith("betterurl")]:
        with open(filename, 'r', encoding='utf8') as f:
            tokens = [t.lower() for t in word_tokenize(f.read()) if t.lower() not in stopwords.words('english')
                      and t.isalpha()]
            token_counts += Counter(tokens)
    for word, count in sorted(token_counts.items(), key=lambda item: item[1], reverse=True)[:25]:
        print(f'Word: {word}, Count: {count}')
    # Chosen words
    words = ['carlsen', 'chess', 'tournament', 'wins', 'championship', 'rapid', 'blitz', 'classical', 'win', 'world']

    df = pd.DataFrame(columns=words, index=list(all_sentences))

    for sentence in all_sentences:
        df.loc[sentence] = [True if word in sentence else False for word in words]
    df = df[df.any(axis=1)]

    print(df.head())

    # DF is our knowledge base, stored efficiently such that each sentence is a row in a table
    # where each word has a True value in that row iff the word exists in the sentence. Therefore
    # when someone wants to know more about a word, all we need to do is return the sentences
    # that have a value of True in that column. Dataframe could also be converted to SQL table
    # if a database connection is given to it.

    df.to_pickle('knowledge_base.p')

    df.to_csv('knowledge_base.csv')

    # end of program
    print("end of crawler")
