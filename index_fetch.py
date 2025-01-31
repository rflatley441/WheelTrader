import requests
import pandas as pd
from bs4 import BeautifulSoup

def SP_fetch():
    url = 'https://www.slickcharts.com/sp500'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  

    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', {'class': 'table table-hover table-borderless table-sm'})

    headers = [header.text.strip() for header in table.find_all('th')]

    rows = []
    for row in table.find_all('tr')[1:]: 
        cols = [col.text.strip() for col in row.find_all('td')]
        if cols:
            rows.append(cols)

    df = pd.DataFrame(rows, columns=headers)

    csv_filename = 'Indexes/S&P500.csv'
    df.to_csv(csv_filename, index=False)

    print(f'Data has been saved to {csv_filename}')

def DOW_fetch():
    url = 'https://www.slickcharts.com/dowjones'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  

    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', {'class': 'table table-hover table-borderless table-sm'})

    headers = [header.text.strip() for header in table.find_all('th')]

    rows = []
    for row in table.find_all('tr')[1:]: 
        cols = [col.text.strip() for col in row.find_all('td')]
        if cols:
            rows.append(cols)

    df = pd.DataFrame(rows, columns=headers)

    csv_filename = 'Indexes/DOWJ.csv'
    df.to_csv(csv_filename, index=False)

    print(f'Data has been saved to {csv_filename}')

def NASDAQ_fetch():
    url = 'https://www.slickcharts.com/nasdaq100'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', {'class': 'table table-hover table-borderless table-sm'})

    headers = [header.text.strip() for header in table.find_all('th')]

    rows = []
    for row in table.find_all('tr')[1:]:  # Skip the header row
        cols = [col.text.strip() for col in row.find_all('td') if col.text.strip()]  # Skip empty columns
        if cols:
            rows.append(cols)

    # Create the DataFrame and ensure it has the correct number of columns
    df = pd.DataFrame(rows, columns=headers)

    # Save the DataFrame to CSV without extra commas
    csv_filename = 'Indexes/NASDAQ1.csv'
    df.to_csv(csv_filename, index=False)

    print(f'Data has been saved to {csv_filename}')

SP_fetch()
DOW_fetch()
NASDAQ_fetch()