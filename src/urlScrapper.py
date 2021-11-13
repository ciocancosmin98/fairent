import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm


IMOBILIARE_FORMAT = 'https://www.imobiliare.ro/inchirieri-apartamente/bucuresti?pagina={0}'

apartmentURLs = []

for i in tqdm(range(450)):
    url = IMOBILIARE_FORMAT.format(i+1)

    try:
        r = requests.get(url)
    except:
        continue

    soup = BeautifulSoup(r.content, 'html.parser')

    for item in soup.find_all('h2', class_='titlu-anunt hidden-xs'):
        apartmentURLs.append(item.contents[1].attrs['href'])

jsonData = json.dumps(apartmentURLs)
with open('../data/urls.json', 'w') as file:
    file.write(jsonData)