from typing import List
import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from copy import deepcopy
import threading

with open('apartmentURLs.json', 'r') as file:
    apartmentURLs = json.load(file)

def split_tasks(tasks: List, n: int):
    l = len(tasks)
    assert l >= n

    bucket_size = l // n
    result: List[List] = []

    for i in range(n):
        result.append(tasks[i * bucket_size : (i+1) * bucket_size])

    end = n * bucket_size
    for i in range(end, l):
        result[i - end].append(tasks[i])

    return result

def gather_all(urls, n_threads = 4):
    apartments = []

    assert n_threads >= 1

    if n_threads > len(urls):
        n_threads = len(urls)

    split_urls = split_tasks(urls, n_threads)
    
    with tqdm(total=len(urls)) as pbar:
        threads = []
        for i in range(n_threads):
            thread = threading.Thread(target=gather_thread, args=(apartments, split_urls[i], pbar))
            thread.start()
            threads.append(thread)
        
        for i in range(n_threads):
            threads[i].join()

    jsonData = json.dumps(apartments, indent=4, separators=(',', ': '), sort_keys=True)
    with open('apartmentData.json'.format(id), 'w') as file:
        file.write(jsonData)

def gather_thread(apartments, apartmentURLs, pbar):
    for aptURL in tqdm(apartmentURLs):
        try:
            r = requests.get(aptURL)
            soup = BeautifulSoup(r.content, 'html.parser')
            chars = soup.find(id='b_detalii_caracteristici')

            apartment = {}
            for chr in chars.find_all('li'):
                # deepcopy to allow the memory to be freed
                name  = deepcopy(str(chr.contents[0][:-1]))
                value = deepcopy(str(chr.contents[1].contents[0]))

                apartment[name] = value

            price = soup.find('div', class_='pret')
            # deepcopy to allow the memory to be freed
            apartment['pret'] = deepcopy(str(price.contents[1]))

            apartments.append(apartment)
        except:
            pass

        pbar.update(1)


gather_all(apartmentURLs, 8)