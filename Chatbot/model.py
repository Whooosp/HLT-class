import unicodedata

import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import gc
import csv


LABELS = ('PERSON_WHO_IS', 'PERSON_WHERE_IS', 'TOURNAMENT_WHO_IS', 'TOURNAMENT_WHERE_IS')


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

        return logits['logits']


class QuestionDataset(Dataset):
    def __init__(self, df, dev, tokenizer):
        self.x = tokenizer(df['Text'].values.tolist(), padding=True,
                           return_tensors='pt', return_token_type_ids=True)
        self.y = [LABELS.index(label) for label in df['Label']]
        self.device = dev

    def __getitem__(self, index):
        return {"input_ids": torch.tensor(self.x['input_ids'][index]).to(self.device),
                "token_type_ids": torch.tensor(self.x['token_type_ids'][index]).to(self.device),
                "attention_mask": torch.tensor(self.x['attention_mask'][index]).to(self.device),
                "labels": torch.tensor(self.y[index]).to(self.device)}

    def __len__(self):
        return len(self.y)


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

        # print(outputs)
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
        # if (i + 1) % 4 == 0 or i + 1 == len(train_loader):
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


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss, correct = 0, 0.
    true_pos, false_neg, false_pos = [[0. for _ in range(len(LABELS))] for _ in range(3)]
    i = 0
    predictions = torch.tensor([]).to(dataloader.dataset.device)
    with torch.no_grad():
        for data in dataloader:
            inputs, t_ids, a_mask, labels = data.values()
            in_data = {"input_ids": inputs,
                       "token_type_ids": t_ids,
                       "attention_mask": a_mask,
                       }
            pred = model(in_data)
            test_loss += loss_fn(pred, labels).item()
            predictions = torch.concat((predictions, pred), 0)
            acc, t_p, f_n, f_p = multi_acc(pred, labels)
            correct += acc
            true_pos = [true_pos[i] + t_p[i] for i in range(len(LABELS))]
            false_neg = [false_neg[i] + f_n[i] for i in range(len(LABELS))]
            false_pos = [false_pos[i] + f_p[i] for i in range(len(LABELS))]
            precision = [true_pos[i] / (true_pos[i] + false_pos[i]) for i in range(len(LABELS))]
            recall = [true_pos[i] / (true_pos[i] + false_neg[i]) for i in range(len(LABELS))]
            if (i + 1) % 50 == 0:
                f1_score = sum(2 * precision[i] * recall[i] / (precision[i] + recall[i])
                               for i in range(len(LABELS))) / len(LABELS)
                print(
                    f'{i + 1} batches: Accuracy: {correct / (i + 1)}, Loss: {test_loss / (i + 1)}, '
                    f'Precision: {precision}, Recall: {recall}, '
                    f'F score: {f1_score}')
            i += 1

    precision = [true_pos[i] / (true_pos[i] + false_pos[i]) for i in range(len(LABELS))]
    recall = [true_pos[i] / (true_pos[i] + false_neg[i]) for i in range(len(LABELS))]
    test_loss /= num_batches
    correct /= num_batches
    f1_score = sum(2 * precision[i] * recall[i] / (precision[i] + recall[i])
                   for i in range(len(LABELS))) / len(LABELS)
    print(
        f"Test error: \nAccuracy: {correct}%, Avg loss: {test_loss}, Precision: {precision}, Recall: {recall}, "
        f"F score: {f1_score}\n")
    return correct, test_loss, predictions


def run_train(model, train_dataset, val_dataset, train_batch, lr, EPOCHS):
    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=1)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_vloss = 1_000_000.

    start_time = time.time()
    best_acc = 0.
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.to(device)
        model.train(True)
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, epoch)
        # We don't need gradients on to do reporting
        model.train(False)

        # del data, in_data

        # model.to('cpu')

        gc.collect()
        torch.cuda.empty_cache()

        running_vloss = 0.0
        with torch.no_grad():
            acc, avg_vloss, _ = test_loop(validation_loader, model, loss_fn)

        # del vdata
        gc.collect()
        # avg_vloss = running_vloss / len(validation_loader)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss or acc > best_acc:
            best_acc = acc
            best_vloss = avg_vloss
            model_path = f'./model_{lr}_{acc}_{epoch}.pt'
            model_path = model_path
            torch.save(model.state_dict(), model_path)

    end_time = time.time()
    print(f'--- {(end_time - start_time) / 3600} hours ---')


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    #
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    #
    bert_model = BertClassifier()
    bert_model.to(device)

    total_df = pd.read_csv('Training_Data.txt', quoting=csv.QUOTE_NONE, sep='|', names=['Text', 'Label'])
    total_df['Text'] = total_df['Text'].apply(
        lambda val: ''.join(unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode()).strip())
    total_df['Label'] = total_df['Label'].apply(lambda val: val.strip())

    train, test = train_test_split(total_df, test_size=0.2)

    train_dataset, test_dataset = QuestionDataset(train, device, tokenizer), QuestionDataset(test, device, tokenizer)

    run_train(bert_model, train_dataset, test_dataset, 16, 3e-5, 3)

    # print(train, test)

    # print(sum(train['Label'] == 'PERSON_WHO_IS') / len(train))
    # X = tokenizer("hello, how are you", padding=True,
    #               return_tensors='pt', return_token_type_ids=True)
    # X['input_ids'] = X['input_ids'].to(device)
    # X['token_type_ids'] = X['token_type_ids'].to(device)
    # X['attention_mask'] = X['attention_mask'].to(device)
    #
    # print(X)
    # print(bert_model(X))
