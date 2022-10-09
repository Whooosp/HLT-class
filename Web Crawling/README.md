# Web Crawling
### [Aditya Guin](https://adityaguin.github.io/CS-4395/) & Varin Sikand

## Overview
Building a Web Crawler to generate a knowledge base for Magnus Carlsen, world renowned chess champion.
Utilizes nltk's sentence and word tokenizer along with beautifulsoup for web scraping and crawling.
Saves knowledge base in a pandas dataframe and exports it as a pickle file and csv file. Also creates
text files for each link scraped, both an unprocessed version (url_NAME.txt) and processed version
(betterurl_NAME.txt). Links are chosen based off of Carlsen's Wikipedia page; using only relevant
links from the page. More details can be found in the "Web Crawler Knowledge Base Report" pdf.

## How to Run
Once downloaded, navigate to this folder and run the following command

**python web_crawler.py**