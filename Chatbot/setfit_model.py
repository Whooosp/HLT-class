import unicodedata

import torch
import pandas as pd
from setfit import SetFitModel, SetFitTrainer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sentence_transformers.losses import CosineSimilarityLoss
import time
import numpy as np
import gc
import csv
from datasets import load_dataset

LABELS = ('PERSON_WHO_IS', 'PERSON_WHERE_IS', 'TOURNAMENT_WHO_IS', 'TOURNAMENT_WHERE_IS')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, dev):
        self.text = df['Text'].values.tolist()
        self.label = [LABELS.index(label) for label in df['Label']]
        self.label_text = df['Label'].tolist()
        self.device = dev
        self.data_dict = {'text': self.text, 'label': self.label, 'label_text': self.label_text}
        self.column_names = self.data_dict.keys()

    def __getitem__(self, index):
        return self.data_dict[index]

    def __len__(self):
        return len(self.label)

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'

    model = SetFitModel.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2',
                                        use_differentiable_head=True,
                                        head_params={'out_features': len(LABELS)})
    #
    #
    total_df = pd.read_csv('Training_Data.txt', quoting=csv.QUOTE_NONE, sep='|', names=['Text', 'Label'])
    total_df['Text'] = total_df['Text'].apply(
        lambda val: ''.join(unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode()).strip())
    total_df['Label'] = total_df['Label'].apply(lambda val: val.strip())

    train, test = train_test_split(total_df, test_size=0.2)

    # dataset = load_dataset('SetFit/SentEval-CR')
    # print(dataset['train'])
    train_dataset, test_dataset = Dataset(train, device), Dataset(test, device)
    # print(train_dataset)
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss_class=CosineSimilarityLoss,
        metric='accuracy',
        batch_size=16,
        num_iterations=20,
        num_epochs=1
    )

    trainer.train(
        num_epochs=3,
        batch_size=16,
        body_learning_rate=3e-5,
        learning_rate=1e-3
    )
    metrics = trainer.evaluate()
    print(metrics)
    # run_train(bert_model, train_dataset, test_dataset, 16, 3e-5, 3)