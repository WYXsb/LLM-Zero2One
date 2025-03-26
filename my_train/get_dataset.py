import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "sms_spam_collection.tsv"

def download_and_extract_data(url,zip_path,extracted_path,data_file_path):
    if data_file_path.exists():
        print("Data already exists, skipping download.")
        return 
    
    with urllib.request.urlopen(url) as response:
        with open(zip_path,'wb') as out_file:
            out_file.write(response.read())
            
    # unzipping the file
    with zipfile.ZipFile(zip_path,'r') as zip_ref:
        zip_ref.extractall(extracted_path)
        
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path,data_file_path)
    print(f"File downloaded and extracted to {data_file_path}")
    
try:
    download_and_extract_data(url,zip_path,extracted_path,data_file_path)
except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
    print(f"Primary URL failed: {e}. Trying backup URL...")
    url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
    download_and_extract_data(url, zip_path, extracted_path, data_file_path) 
    
    
    