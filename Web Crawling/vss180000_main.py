import requests
import re
from bs4 import BeautifulSoup


def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True


def scrape_url(url: str, to_save: bool):
    r = requests.get(url)

    data = r.text
    soup = BeautifulSoup(data, 'html.parser')

    data = soup.find_all(text=True)

    titles = [t for t in data if t.parent.name == "title"]
    title = titles[0] if len(titles) > 0 else url[url.index('://')+3:].replace('/', '_')
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
        if ('Carlsen' in link_str or 'carlsen' in link_str) and 'chessbomb' not in link_str:
            if link_str.startswith('/url?q='):
                link_str = link_str[7:]
            if '&' in link_str:
                i = link_str.find('&')
                link_str = link_str[:i]
            if link_str.startswith('http') and 'google' not in link_str:
                urls.add(link_str)

        if len(urls) > 14:
            break

    for url in urls:
        print(url)
        scrape_url(url, True)
    # for soup in
    # end of program
    print("end of crawler")
