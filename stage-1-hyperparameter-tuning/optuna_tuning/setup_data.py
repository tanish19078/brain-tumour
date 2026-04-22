import zipfile
import os

def setup_data():
    if not os.path.exists('dataset'):
        print('Unzipping dataset...')
        if not os.path.exists('dataset.zip'):
            print("Error: dataset.zip not found!")
            return
            
        with zipfile.ZipFile('dataset.zip', 'r') as z:
            z.extractall('.')
        print('Dataset unzipped successfully.')
    else:
        print('Dataset directory already exists. Skipping unzip.')

if __name__ == '__main__':
    setup_data()
