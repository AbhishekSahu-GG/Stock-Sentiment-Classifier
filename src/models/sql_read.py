
# import torch
# import transformers
# import pandas as pd
# import numpy as np
# import mlflow
# import mlflow.pytorch
# from torch.utils.data import Dataset, DataLoader
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# from torch.optim import AdamW
# from sklearn.metrics import classification_report, f1_score
# print('all imports ok')
from huggingface_hub import HfApi

api = HfApi()
api.create_repo('stock-sentiment-distilbert', private=False)
api.upload_folder(
    folder_path='models/distilbert',
    repo_id='MightyEagle1/stock-sentiment-distilbert',
    repo_type='model'
)
print('Done')