import urllib.request
from bs4 import BeautifulSoup

url = 'https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')

soup = BeautifulSoup(html, 'html.parser')
links = soup.find_all('a')
for link in links:
    text = link.get_text().strip()
    if 'download' in text.lower() or 'login' in text.lower() or 'log in' in text.lower() or 'access' in text.lower():
        print(f"Link: {text}, href: {link.get('href')}")
