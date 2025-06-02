import json
import os
import sys
import requests

# Constants
BASE_URL = "https://kask.eti.pg.edu.pl/cam"
SAVE_DIR = os.path.join("./../../", "data")
session = requests.Session()


def fetch_data() -> dict:
    """Fetch all training data IDs and their corresponding data."""
    all_url = f"{BASE_URL}/training_data/all"
    response = session.get(all_url)
    result = {}

    id_index = 0
    while (id_index := response.text.find('Id: ', id_index + 1)) != -1:
        id_end = response.text.find('</h2>', id_index)
        data_id = response.text[id_index + 4:id_end]
        result[data_id] = fetch_datapoint(data_id)
    return result


def fetch_datapoint(datapoint_id: str) -> dict:
    """Fetch details of a specific training data point by ID."""
    url = f"{BASE_URL}/training_data/{datapoint_id}"
    response = session.get(url)

    def extract_value(key: str) -> str:
        start = response.text.find(key)
        end = response.text.find('</span>', start)
        return response.text[start + len(key) + 2:end]

    def is_float(value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False

    def is_int(value: str) -> bool:
        try:
            int(value)
            return True
        except ValueError:
            return False

    img_start = response.text.find('<img src="/cam/training_data/photos/')
    img_end = response.text.find('"', img_start + 35)
    img = os.path.basename(response.text[img_start + 35:img_end])

    total_area = extract_value('Total area')
    total_length = extract_value('Total length')
    mean_thickness = extract_value('Mean thickness')
    branching_points = extract_value('Branching points')

    # Validate numeric values
    if not all(is_float(value) for value in [total_area, total_length, mean_thickness, branching_points]):
        return {}

    return {
        'id': datapoint_id,
        'created_at': extract_value('Created at'),
        'total_area': float(total_area),
        'total_length': float(total_length),
        'mean_thickness': float(mean_thickness),
        'branching_points': float(branching_points),
        'is_good': extract_value('Is the tissue good') != 'N/a',
        'scale': int(extract_value('Photo scale')) if is_int(extract_value('Photo scale')) else 0,
        'img': img
    }


def login(key: str) -> bool:
    """Authenticate and set session cookie."""
    form = session.get(f"{BASE_URL}/auth/login")
    if not form:
        return False

    csrf_start = form.text.find('value', form.text.find('csrf_token')) + 7
    csrf_token = form.text[csrf_start:form.text.find('"', csrf_start)]

    response = session.post(
        f"{BASE_URL}/auth/login",
        data={"key": key, 'submit': 'Login', 'csrf_token': csrf_token}
    )
    return 'Invalid key' not in response.text


def download_photos(data: dict) -> None:
    """Download photos from the data."""
    photo_url = f"{BASE_URL}/training_data/photos/"
    os.makedirs(os.path.join(SAVE_DIR, "photos"), exist_ok=True)

    for item in data.values():
        if not item:
            continue
        img_url = f"{photo_url}{item['img']}"
        response = session.get(img_url, stream=True)
        if response.status_code == 200:
            with open(os.path.join(SAVE_DIR, "photos", item['img']), "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Downloaded: {item['img']}")


def main():
    """Main execution function."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    if len(sys.argv) < 2:
        print("No key provided.")
        return

    if not login(sys.argv[1]):
        print("ERROR: Authentication failed.")
        return

    print("Fetching data...")
    data = fetch_data()

    with open(os.path.join(SAVE_DIR, "data.json"), 'w') as file:
        json.dump(data, file, indent=4)

    print("Downloading photos...")
    download_photos(data)


if __name__ == "__main__":
    main()
