import json
import os
import sys
import requests

# Constants
BASE_URL = "https://kask.eti.pg.edu.pl/cam"
SAVE_DIR = os.path.join("./../../", "data")

# Initialize a session
session = requests.Session()


def fetch_data() -> dict:
    """
    Fetch all training data IDs and their corresponding data.
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
    Fetch details of a specific training data point by its ID.
    """
    datapoint_url = BASE_URL + '/training_data/'
    datapoint_resp = session.get(datapoint_url + datapoint_id)

    img_start = datapoint_resp.text.find('<img src="/cam/training_data/photos/')
    img_end = datapoint_resp.text.find('"', img_start + len('<img src="/cam/training_data/photos/'))
    img_full = datapoint_resp.text[img_start + len('<img src="/cam/training_data/photos/'):img_end]
    img = os.path.basename(img_full)

    # Scrape data point properties
    created_at = scrape_datapoint_value(datapoint_resp.text, 'Created at')
    total_area = scrape_datapoint_value(datapoint_resp.text, 'Total area')
    total_length = scrape_datapoint_value(datapoint_resp.text, 'Total length')
    mean_thickness = scrape_datapoint_value(datapoint_resp.text, 'Mean thickness')
    branching_points = scrape_datapoint_value(datapoint_resp.text, 'Branching points')
    is_good = scrape_datapoint_value(datapoint_resp.text, 'Is the tissue good')
    scale = scrape_datapoint_value(datapoint_resp.text, 'Photo scale')

    # Helper to validate float conversion
    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    # Validate numerical properties
    if not all(is_float(value) for value in [total_area, total_length, mean_thickness, branching_points]):
        return {}

    # Build result dictionary
    result = {
        'id': datapoint_id,
        'created_at': created_at,
        'total_area': float(total_area),
        'total_length': float(total_length),
        'mean_thickness': float(mean_thickness),
        'branching_points': float(branching_points),
        'is_good': is_good != 'N/a' and bool(is_good),
        'scale': int(scale) if scale != 'N/a' else 0,
        'img': img
    }

    return result


def scrape_datapoint_value(body: str, value_key: str) -> str:
    """
    Extract the value of a specific property from the HTML content.
    """
    start = body.find(value_key)
    end = body.find('</span>', start)
    return body[start + len(value_key) + len(': '):end]


def login(key: str) -> bool:
    """
    Authenticate with the provided key and set the session cookie.
    """
    LOGIN_SUB_URL = "/auth/login"
    form = session.get(BASE_URL + LOGIN_SUB_URL)
    if form is None:
        raise False

    if not form:
        return False

    # Extract CSRF token
    form_str = form.text
    csrf_token_start = form_str.find('value', form_str.find('csrf_token')) + len('value="')
    csrf_token = form_str[csrf_token_start:form_str.find('"', csrf_token_start)]

    # Submit login form
    response = session.post(BASE_URL + LOGIN_SUB_URL, data={
        "key": key,
        'submit': 'Login',
        'csrf_token': csrf_token
    })

    return 'Invalid key' not in response.text


def download_photos(data: dict) -> None:
    """
    Download photos associated with the training data.
    """
    photo_url = BASE_URL + '/training_data/photos/'
    photos_dir = os.path.join(SAVE_DIR, "photos")
    os.makedirs(photos_dir, exist_ok=True)

    for single_data in data.values():
        if not single_data:
            continue

        img_url = photo_url + single_data['img']
        response = session.get(img_url, stream=True)

        if response.status_code == 200:
            filename = os.path.join(photos_dir, os.path.basename(img_url))
            with open(filename, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {filename}")


def main():
    """
    Main function to execute the scraping workflow.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        key = sys.argv[1]
    except IndexError:
        print("No key provided.")
        return

    if not login(key):
        print("ERROR: Authentication failed.")
        return

    print("Logged in. Fetching data...\n")
    data = fetch_data()

    # Save data as JSON
    json_path = os.path.join(SAVE_DIR, "data.json")
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)

    print("Data fetched. Downloading photos...\n")
    download_photos(data)


if __name__ == "__main__":
    main()
