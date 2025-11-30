import os
import io
import zipfile
import requests
import numpy as np

def get_glove():
    # 2. Set destination folder in Drive
    glove_drive_dir = "/content/drive/MyDrive/glove_6B/"
    os.makedirs(glove_drive_dir, exist_ok=True)

    zip_path = glove_drive_dir + "glove.6B.zip"

    # 3. Download only if not already stored in Drive
    if not os.path.exists(zip_path):
        print("Downloading GloVe embeddings (glove.6B.zip)...")
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        r = requests.get(url, stream=True)

        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    else:
        print("GloVe zip already exists in Drive. Skipping download.")

    # 4. Extract only if not already extracted
    extract_flag = not any(fname.startswith("glove.6B.") and not fname.endswith("zip") for fname in os.listdir(glove_drive_dir))

    if extract_flag:
        print("Extracting GloVe files...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(glove_drive_dir)
        print("Extraction complete.")
    else:
        print("GloVe already extracted. Skipping extraction.")

    # 5. List available files
    print("\nFiles stored in your Drive GloVe directory:")
    print(os.listdir(glove_drive_dir))  


def get_fasttext():
    # 2. Set destination folder in Drive
    drive_dir = "/content/drive/MyDrive/fasttext/"
    os.makedirs(drive_dir, exist_ok=True)

    zip_path = drive_dir + "wiki-news-300d-1M-subword.vec.zip"

    # 3. Download only if not already stored in Drive
    if not os.path.exists(zip_path):
        print("Downloading fasttext embeddings ...")
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip"
        r = requests.get(url, stream=True)

        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    else:
        print("Zip already exists in Drive. Skipping download.")

    # 4. Extract only if not already extracted
    extract_flag = not any(fname.startswith("wiki") and not fname.endswith("zip") for fname in os.listdir(drive_dir))

    if extract_flag:
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(drive_dir)
        print("Extraction complete.")
    else:
        print("Fasttext already extracted. Skipping extraction.")

    # 5. List available files
    print("\nFiles stored in your Drive Fasttext directory:")
    
    print(os.listdir(drive_dir))  
    
    fname = "wiki-news-300d-1M-subword.vec"
    fin = io.open(os.path.join(drive_dir, fname), 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data
