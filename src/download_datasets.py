import gdown
import os
import pandas as pd

csv_path = {
    "Base": "https://drive.google.com/file/d/1tx_oAk_tkoxOAG0bD0RSagSUPONTcBI-/view?usp=sharing",
    "X_train_resampled_featureeng": "https://drive.google.com/file/d/1NqCdZ7wuIJwapKaHAIyvNfrwskLXWhyI/view?usp=sharing",
    "y_train_resampled_featureeng": "https://drive.google.com/file/d/10W4Dk_7lcv0Yuht7KfIs_2m85VlaNVs8/view?usp=sharing",
    "X_train_resampled_nonfeatureeng": "https://drive.google.com/file/d/1q9Yv9jPklCEqb0uOQoOf6C9e2nZjx9QU/view?usp=sharing",
    "y_train_resampled_nonfeatureeng": "https://drive.google.com/file/d/1N5fhZzvter4hnp6sND8erTM4XJS93p4R/view?usp=sharing"
}

def download_datasets():
    try: 
        for key, value in csv_path.items():
            file_id = value.split('/')[-2]  # Extracted from the URL
            output = f"./data/{key}.csv"

            if os.path.exists(os.path.join(os.getcwd(), output)):
                print(f"{key}.csv already exists!")
                continue

            gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", output, quiet=False) # quiet = show progress bar

            print(f"{key}.csv successfully downloaded!")

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False