import requests
from bs4 import BeautifulSoup

# URL to scrape
url = "https://celestinniyomugabo.github.io/Python-Lab"
url = "https://agasobanuyefilms.com"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
links = soup.find_all('button')

# for p in paragraphs:
#     print(p.text.strip())

print(links)


# Webscrapping from a file
content = open("index.html", "r").read()
soup = BeautifulSoup(content, "html.parser")
paragraphs = soup.find_all('p')
links = soup.find_all("a")
headings = soup.find_all("h2")
headings

divs = soup.find_all("div")
divs = [div.get_text(strip=True) for div in divs]
divs