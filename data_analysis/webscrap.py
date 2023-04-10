import requests
from bs4 import BeautifulSoup

pages = 'https://www.bgmringtones.com/category/old-bgm-collections/'
response = requests.get(pages)
soup = BeautifulSoup(response.content, 'html.parser')
page_numbers = soup.find_all('a', class_='page-numbers')
urls = []
url = 'https://www.bgmringtones.com/category/old-bgm-collections/'
urls.append(url)
for page in page_numbers:
    href = page['href']
    urls.append(href)
print(urls)
download_linki = []

def fetch_url(url, spec):
    response = requests.get(url)

    # parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # find all the <a> tags on the page
    links = soup.find_all('a')

    # print the href attribute value for each <a> tag
    download_links = []
    for link in links:
        # print(link.get('href'))
        d_link = link.get('href')
        # print(d_link)
        if spec in str(d_link):
            download_links.append(d_link)
    return download_links

def url_table(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    table = soup.find('table')
    if table is not None:
        rows = table.find_all('tr')
        hrefs = []
        for row in rows:
            cells = row.find_all('td')
            if len(cells) > 0:
                href = cells[0].find('a')['href']
                hrefs.append(href)
        return hrefs


for url in urls:
    linki = fetch_url(url, "#respond")
    for linka in linki:
        download_linki.append(url_table(linka))
    print("----------")
    count = 0
    for down in download_linki:
        if down is not None:
            count = count + len(down)
            for d in down:
                response = requests.get(d)
                ddd= d.split('=')[-1]
                if ".mp3" in ddd:
                    with open(d.split('=')[-1], 'wb') as f:
                        f.write(response.content)




