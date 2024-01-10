import os
import urllib.request
import zipfile

# URL of the horse-or-human.zip file
zip_url = 'https://storage.googleapis.com/learning-datasets/horse-or-human.zip'

# Assuming horse-or-human.zip is in the same directory as the script
script_directory = os.path.dirname(os.path.abspath(__file__))
local_zip = os.path.join(script_directory, 'horse-or-human.zip')

# Download the zip file
print(f'Downloading {zip_url}...')
urllib.request.urlretrieve(zip_url, local_zip)
print('Download complete.')

# Extract the contents of the zip file
extracted_dir = os.path.join(script_directory, 'horse-or-human')
print(f'Extracting contents to {extracted_dir}...')
with zipfile.ZipFile(local_zip, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)
print('Extraction complete.')

# Optional: Remove the downloaded zip file after extraction
os.remove(local_zip)

# Directory with our training horse pictures
train_horse_dir = os.path.join(extracted_dir, 'horses')

# Directory with our training human pictures
train_human_dir = os.path.join(extracted_dir, 'humans')

# Print the first 10 filenames in each directory
print('First 10 training horse images:', os.listdir(train_horse_dir)[:10])
print('First 10 training human images:', os.listdir(train_human_dir)[:10])

# Print the total number of images in each directory
print('Total training horse images:', len(os.listdir(train_horse_dir)))
print('Total training human images:', len(os.listdir(train_human_dir)))

