import csv
import requests
from bs4 import BeautifulSoup

url = "https://www.travelhackingtool.com/airport/"

response = requests.get(url)
html_content = response.content

soup = BeautifulSoup(html_content, 'html.parser')

# Trouver le tableau contenant les codes IATA des aéroports
table = soup.find('table')

airport_data = []

# Parcourir les lignes du tableau
for row in table.find_all('tr'):
    # Extraire les données de chaque colonne
    cols = row.find_all('td')
    iata = cols[0].text.strip()
    name = cols[1].text.strip()
    country = cols[2].text.strip()
    airport_data.append([iata, name, country])


# Écrire les données dans un fichier CSV
with open('airport_codes.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Airport Name', 'IATA Code', 'Country'])
    writer.writerows(airport_data)
