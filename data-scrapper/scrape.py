import json
import os
import sys
import requests
import datetime
from urllib.parse import urljoin

BASE_URL = "https://kask.eti.pg.edu.pl/cam"
# BASE_URL = "http://127.0.0.1:5000/cam"
SAVE_DIR = f"./data/scraped-{datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}/"

session = requests.Session()

def fetch_data() -> dict:
    """
    Fetch training data.
    """
    result = {}

    all_url = BASE_URL + '/training_data/all'
    all_response = session.get(all_url)
    id_index = 0
    while True:
        id_index = all_response.text.find('Id: ', id_index + 1)
        if id_index == -1:
            break
        id_end = all_response.text.find('</h2>', id_index)
        data_id = all_response.text[id_index + len('Id: '):id_end]

        result[data_id] = fetch_datapoint(data_id)
    return result


def fetch_datapoint(datapoint_id: str) -> dict:
    """
    Fetch datapoint data.
    """
    datapoint_url = BASE_URL + '/training_data/'
    datapoint_resp = session.get(datapoint_url + datapoint_id)

    img_start = datapoint_resp.text.find('<img src="/cam/training_data/photos/')
    img_end = datapoint_resp.text.find('"', img_start + len('<img src="/cam/training_data/photos/'))
    img_full = datapoint_resp.text[img_start + len('<img src="/cam/training_data/photos/'):img_end]
    img = os.path.basename(img_full)

    result = {
        'id': datapoint_id,
        'created_at': scrape_datapoint_value(datapoint_resp.text, 'Created at'),
        'total_area': float(scrape_datapoint_value(datapoint_resp.text, 'Total area')),
        'total_length': float(scrape_datapoint_value(datapoint_resp.text, 'Total length')),
        'mean_thickness': float(scrape_datapoint_value(datapoint_resp.text, 'Mean thickness')),
        'branching_points': float(scrape_datapoint_value(datapoint_resp.text, 'Branching points')),
        'is_good': scrape_datapoint_value(datapoint_resp.text, 'Is the tissue good'),
        'scale': scrape_datapoint_value(datapoint_resp.text, 'Photo scale'),
        'img': img
    }

    if result['is_good'] == 'N/a':
        result['is_good'] = False
    else:
        result['is_good'] = bool(result['is_good'])

    if result['scale'] == 'N/a':
        result['scale'] = 0
    else:
        result['scale'] = int(result['scale'])

    return result


def scrape_datapoint_value(body: str, value_key: str) -> any:
    """
    Searches and return value of datapoint property. Assumes the key exists.
    Example key: Total area
    """
    start = body.find(value_key)
    end = body.find('</span>', start)
    return body[start + len(value_key) + len(': '):end]


def login(key: str) -> bool:
    """
    Login and set cookie
    """
    LOGIN_SUB_URL = "/auth/login"
    form = session.get(BASE_URL + LOGIN_SUB_URL)
    if form is None:
        raise False

    # Scrapping the csrf token
    form_str = form.text
    csrf_token_start = form_str.find('value', form_str.find('csrf_token')) + len('value="')
    csrf_toekn = form_str[csrf_token_start:form_str.find('"', csrf_token_start)]

    response = session.post(BASE_URL + LOGIN_SUB_URL, data={"key": key, 'submit': 'Login', 'csrf_token': csrf_toekn})
    if response.text.find('Invalid key') != -1:
        return False
    return True


def download_photos(data) -> None:
    """
    Download photos from the website and save them to the photo directory.
    """
    photo_url = BASE_URL + '/training_data/photos/'
    for single_data in data.values():
        url = photo_url + single_data['img']
        response = session.get(url, stream=True)
        if response.status_code != 200:
            continue
        response.raise_for_status()
        filename = os.path.join(SAVE_DIR + 'photos', os.path.basename(url))

        with open(filename, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded: {filename}")


def main():
    os.makedirs(SAVE_DIR + 'photos/', exist_ok=True)

    try:
        key = sys.argv[1]
    except IndexError:
        print("No key provided")
        return

    if not login(key):
        print('\n\nERROR: Authentication failed')
        return

    print('Logged in. Starting downloading data...\n')
    data = fetch_data()
    with open(SAVE_DIR + 'data.json', 'w') as file:
        json.dump(data, file, indent=4)

    print('Downloaded data. Starting downloading photos...\n')
    download_photos(data)


if __name__ == "__main__":
    main()
