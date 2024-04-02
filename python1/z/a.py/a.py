import requests
import pandas as pd
from bs4 import BeautifulSoup
url = "https://www.jagranjosh.com/general-knowledge/list-of-states-in-usa-1663078166-1"
data = requests.get(url).text
print(data)
# Creating BeautifulSoup object
soup = BeautifulSoup(data, 'html.parser')
# Verifying tables and their classes
print('Classes of each table:')
for table in soup.find_all('table'):
    print(table.get('class'))
# Creating list with all tables
tables = soup.find_all('table')
print(table)

#  Looking for the table with the classes 'wikitable' and 'sortable'
table = soup.find('table', class_='TableData')
# Defining of the dataframe
df = pd.DataFrame(columns=["S No."," US States","US Capital"])
print(df)
