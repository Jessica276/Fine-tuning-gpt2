import pandas as pd
import gdown
import csv
import chardet
import sys
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    url = "https://drive.google.com/uc?id=1-QyZJXzDcwZ3ZqIBk33l6WYXb_OHM51S"
    output = "rakitra.csv"
    gdown.download(url,output)

    csv.field_size_limit(sys.maxsize)

    with open(output, "r", encoding="utf-8") as inf, open("test.csv", "w", encoding="utf-8", newline='') as outf:
        # newline='' to avoid extra blank rows in the output file
        reader = csv.reader(inf)
        writer = csv.writer(outf)

        for row in reader:
            if len(row) > 0:  # Ignore les lignes vides
                writer.writerow(row)

    try:
        data = pd.read_csv("test.csv", encoding="utf-8")  # First try utf-8
    except UnicodeDecodeError:
        data = pd.read_csv("test.csv", encoding="latin-1")  # Fallback to latin-1 if utf-8 fails


    # Split data into train and validation
    text_train, text_val = train_test_split(data, test_size=0.2, random_state=42)

    return text_train, text_val