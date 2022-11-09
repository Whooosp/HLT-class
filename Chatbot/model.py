import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gc

LABELS = ('chess', 'tournament', 'rules', 'players')


class BertClassifier(torch.nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()

        self.encoder = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-large", num_labels=4,
                                                                          problem_type="Single_label_classification")

    def forward(self, x):
        print(x['input_ids'].get_device())
        logits = self.encoder(input_ids=x['input_ids'],
                              token_type_ids=x['token_type_ids'],
                              attention_mask=x['attention_mask'])

        return logits


class QuestionDataset(Dataset):
    def __init__(self, df, dev, tokenizer):
        self.x = tokenizer(df['text'].values.tolist(), padding=True,
                           return_tensors='pt', return_token_type_ids=True)
        self.y = [LABELS.index(label) for label in df['label']]
        self.device = dev

    def __getitem__(self, index):
        return {"input_ids": torch.tensor(self.x['input_ids'][index]).to(self.device),
                "token_type_ids": torch.tensor(self.x['token_type_ids'][index]).to(self.device),
                "attention_mask": torch.tensor(self.x['attention_mask'][index]).to(self.device),
                "labels": torch.tensor(self.y[index]).to(self.device)}


def multi_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.nn.functional.softmax(y_pred, dim=1))

    correct_results_sum = sum(y_pred_tag[i][y_test[i]] == 1 for i in range(len(y_test)))
    precision = 0.
    recall = 0.

    true_pos = [sum(y_pred_tag[i][lbl] == 1 and y_test[i] == lbl for i in range(len(y_test))) for lbl in
                range(len(LABELS))]
    false_neg = [sum(y_pred_tag[i][lbl] != 1 and y_test[i] == lbl for i in range(len(y_test))) for lbl in
                 range(len(LABELS))]
    false_pos = [sum(y_pred_tag[i][lbl] == 1 and y_test[i] != lbl for i in range(len(y_test))) for lbl in
                 range(len(LABELS))]

    # print(y_pred_tag, y_test, correct_results_sum, sep='\n')
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc, true_pos, false_neg, false_pos


def train_one_epoch(model, train_loader, optimizer, loss_fn, epoch_index):
    running_loss = 0.
    last_loss = 0.
    train_acc = 0.
    total_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair

        # print(i, data)
        inputs, t_ids, a_mask, labels = data.values()

        # Zero your gradients for every batch!

        in_data = {"input_ids": inputs,
                   "token_type_ids": t_ids,
                   "attention_mask": a_mask
                   }

        # Make predictions for this batch
        outputs = model(in_data)

        acc, t_p, f_n, f_p = multi_acc(outputs, labels)

        train_acc += acc

        # print(train_acc)
        # print(outputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss = loss / 4
        loss.backward()

        # print(loss)
        # print("BEFORE:\n -------------------------------------")
        # for param in model.parameters():
        #   print(param)
        # # Adjust learning weights
        if (i + 1) % 4 == 0 or i + 1 == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()

        # print("AFTER:\n -------------------------------------")
        # for param in model.parameters():
        #   print(param)

        # Gather data and report
        running_loss += loss.item()
        total_loss += loss.item() * 4
        if i % 100 == 99:
            last_loss = running_loss / 25  # loss per batch
            total_acc = train_acc / 100
            print('  batch {} loss: {} train_acc {}'.format(i + 1, last_loss, total_acc))
            tb_x = epoch_index * len(train_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
            train_acc = 0.
        del in_data, inputs, t_ids, a_mask, labels
        gc.collect()
        torch.cuda.empty_cache()
        # break

    return total_loss / len(train_loader)


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

    bert_model = BertClassifier()
    bert_model.to(device)

    X = tokenizer("hello, how are you", padding=True,
                  return_tensors='pt', return_token_type_ids=True)
    X['input_ids'] = X['input_ids'].to(device)
    X['token_type_ids'] = X['token_type_ids'].to(device)
    X['attention_mask'] = X['attention_mask'].to(device)

    print(X)
    print(bert_model(X))
