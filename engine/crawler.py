import argparse
import json
import os
from html.parser import HTMLParser
from urllib import parse
from urllib.request import urlopen

import html2text


class LinkParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.links = None
        self.base_url = None

    def error(self, message):
        return message

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for (key, value) in attrs:
                if key == 'href':
                    new_url = parse.urljoin(self.base_url, value)
                    self.links = self.links + [new_url]

    def get_links(self, url):
        self.links = []
        self.base_url = url
        response = urlopen(url)
        if 'text/html' in response.getheader('Content-Type'):
            html_bytes = response.read()
            html_string = html_bytes.decode("utf-8")
            self.feed(html_string)
            return html_string, self.links
        else:
            return "", []


def spider(pages_to_visit, destination, max_pages):
    number_visited = 0
    os.makedirs(destination, exist_ok=True)

    while number_visited < max_pages and pages_to_visit != []:
        url = pages_to_visit.pop(0)
        if '#' in url:
            continue
        number_visited += 1
        try:
            print(number_visited, "Visiting:", url)
            parser = LinkParser()
            data, links = parser.get_links(url)

            with open(os.path.join(destination, '%d.txt' % number_visited), 'w') as file:
                data = html2text.html2text(data)
                file.write(data)

            pages_to_visit.extend(links)
        except Exception:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    config = json.load(open(args.config))
    spider(config['pages'], config['destination'], config['max_pages'])
