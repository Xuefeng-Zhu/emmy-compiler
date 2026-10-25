#!/usr/bin/env python3
"""Check local documentation links, source-line anchors and offline page dependencies."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []; self.ids = set(); self.dependencies = []; self.csp = ''
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            if a['id'] in self.ids: raise ValueError('duplicate HTML id: ' + a['id'])
            self.ids.add(a['id'])
        if tag == 'a' and 'href' in a: self.links.append(a['href'])
        if tag in ('script', 'img', 'iframe', 'source') and a.get('src'): self.dependencies.append(a['src'])
        if tag == 'link' and a.get('href'): self.dependencies.append(a['href'])
        if tag == 'meta' and a.get('http-equiv', '').lower() == 'content-security-policy': self.csp = a.get('content', '')


def headings(path):
    result = set(); seen = {}
    for line in path.read_text().splitlines():
        if not re.match(r'^#{1,6} ', line): continue
        text = re.sub(r'^#{1,6} +', '', line).strip().lower()
        slug = re.sub(r'[^\w\- ]', '', text).replace(' ', '-')
        count = seen.get(slug, 0); seen[slug] = count + 1
        result.add(slug + (f'-{count}' if count else ''))
    return result


def check_link(source, link):
    target = urlsplit(link)
    if target.scheme or target.netloc: return
    path = (source.parent / unquote(target.path)).resolve() if target.path else source
    if not path.is_relative_to(ROOT) or not path.exists(): raise ValueError(f'{source.relative_to(ROOT)}: missing/outside link {link}')
    fragment = unquote(target.fragment)
    if not fragment: return
    if path.suffix == '.py' and re.fullmatch(r'L\d+(?:-L\d+)?', fragment):
        for line in re.findall(r'L(\d+)', fragment):
            if not 1 <= int(line) <= len(path.read_text().splitlines()): raise ValueError(f'bad source line: {link}')
    elif path.suffix == '.md' and fragment not in headings(path):
        raise ValueError(f'{source.relative_to(ROOT)}: missing heading {link}')
    elif path.suffix == '.html':
        page = Page(); page.feed(path.read_text())
        if fragment not in page.ids: raise ValueError(f'missing HTML anchor: {link}')


def main():
    files = [ROOT/'README.md', ROOT/'THIRD_PARTY_NOTICES.md', ROOT/'docs/explainer/README.md'] + sorted((ROOT/'docs').glob('*.md'))
    links = 0
    for path in files:
        text = re.sub(r'```[\s\S]*?```', '', path.read_text())
        for link in re.findall(r'\]\(([^\s)]+)\)', text):
            check_link(path, link); links += 1
    path = ROOT/'docs/explainer/index.html'; text = path.read_text(); page = Page(); page.feed(text)
    for link in page.links: check_link(path, link); links += 1
    if page.dependencies: raise ValueError('offline page contains external/subresource dependencies')
    if "connect-src 'none'" not in page.csp: raise ValueError('offline page must block network connections')
    if re.search(r'\b(?:fetch\s*\(|XMLHttpRequest|WebSocket|sendBeacon|localStorage|sessionStorage)\b', text):
        raise ValueError('offline page contains a network/storage API')
    if re.search(r'url\s*\(', text, re.I): raise ValueError('offline page CSS must not load resources')
    match = re.search(r'<script id="emmy-data" type="application/json">(.*?)</script>', text, re.S)
    data = json.loads(match[1]); assert len(data['exp']['rows']) == 4097
    print(f'PASS: {len(files)} Markdown documents, {links} links, source/heading anchors, and offline HTML dependency checks.')
    return 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (ValueError, OSError, AssertionError) as exc:
        print(f'check_docs.py: {exc}', file=sys.stderr); raise SystemExit(1)
