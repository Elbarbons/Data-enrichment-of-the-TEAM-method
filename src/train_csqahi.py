import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from models import Model
from transformers.optimization import get_scheduler
from sklearn.metrics import accuracy_score, f1_score


# Initialize the dataset
class csqahiDataset(Dataset):
    def __init__(self, filename, sep_token, shuffle, percent=100):
        content, labels = [], []
        df = pd.read_json(filename)

        if shuffle:
            df = df.sample(frac=1).reset_index()

        if filename == 'Datasets/csqa/csqahi-train.json':
            max = round((len(df)*int(percent))/100)
            df = df.iloc[0:max]

        for j in range(len(df)):
            row = df.iloc[j]
            if "answer" in row:
                l = list(row['options']).index(row["answer"])
            else:
                l = 0

            a1, a2, a3, a4 = row['options'][0], row['options'][1], row['options'][2], row['options'][3]
            c, q = row["out_of_context_options"], row["question"]

            content.append("Question: {} {} Answer: {}".format(q, sep_token, a1))
            content.append("Question: {} {} Answer: {}".format(q, sep_token, a2))
            content.append("Question: {} {} Answer: {}".format(q, sep_token, a3))
            content.append("Question: {} {} Answer: {}".format(q, sep_token, a4))

            if l == 0:
                labels += [1, 0, 0, 0]
            elif l == 1:
                labels += [0, 1, 0, 0]
            elif l == 2:
                labels += [0, 0, 1, 0]
            elif l == 3:
                labels += [0, 0, 0, 1]

        self.content, self.labels = content, labels

    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):
        s1, s2 = self.content[index], self.labels[index]
        return s1, s2

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]

# Prepare dataloaders
def configure_dataloaders(sep_token, train_batch_size=16, eval_batch_size=16, shuffle=False, percent=100):

    train_dataset = csqahiDataset("Datasets/csqa/csqahi-train.json", sep_token,
                                     True, percent)
    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=train_batch_size,
                              collate_fn=train_dataset.collate_fn)

    val_dataset = csqahiDataset("Datasets/csqa/csqahi-val.json", sep_token,
                                  False)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=eval_batch_size, collate_fn=val_dataset.collate_fn)

    test_dataset = csqahiDataset("Datasets/csqa/csqahi-test.json", sep_token, False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=eval_batch_size, collate_fn=val_dataset.collate_fn)

    return train_loader, val_loader, test_loader

#  Prepare optimizer and schedule (linear warmup and decay)
def configure_optimizer(model, args):

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    return optimizer

# Prepare scheduler
def configure_scheduler(optimizer, num_training_steps, args):

    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else math.ceil(num_training_steps * args.warmup_ratio)
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler


def train_or_eval_model(model, dataloader, optimizer=None, split="Train"):
    losses, preds, preds_cls, labels_cls, = [], [], [], []
    if split == "Train":
        model.train()
    else:
        model.eval()

    for batch in tqdm(dataloader, leave=False):
        if split == "Train":
            optimizer.zero_grad()

        content, l_cls = batch
        loss, p, p_cls = model(batch)

        preds.append(p)
        preds_cls.append(p_cls)
        labels_cls.append(l_cls)

        if split == "Train":
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    avg_loss = round(np.mean(losses), 4)

    all_preds_cls = [item for sublist in preds_cls for item in sublist]
    all_labels_cls = [item for sublist in labels_cls for item in sublist]
    acc = round(accuracy_score(all_labels_cls, all_preds_cls), 4)
    f1 = round(f1_score(all_labels_cls, all_preds_cls, average="macro"), 4)

    instance_preds = [item for sublist in preds for item in sublist]
    instance_labels = np.array(all_labels_cls).reshape(-1, args.num_choices).argmax(1)
    instance_acc = round(accuracy_score(instance_labels, instance_preds), 4)

    return avg_loss, acc, instance_acc, f1


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for transformers.")
    parser.add_argument("--wd", default=0.1, type=float, help="Weight decay for transformers.")
    parser.add_argument("--warm-up-steps", type=int, default=0, help="Warm up steps.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
    parser.add_argument("--bs", type=int, default=4, help="Batch size.")
    parser.add_argument("--eval-bs", type=int, default=4, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--name", default="xlm-roberta-base", help="Which model.")
    parser.add_argument("--n-choices", type=int, default=4, help="Num choices in an instance.")
    parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle train data such that positive and negative \
        sequences of the same question are not necessarily in the same batch.")
    parser.add_argument('--subsample', default=100, help="percent of the training dataset to use")

    global args
    args = parser.parse_args()
    print(args)

    percent = args.subsample
    train_batch_size = args.bs
    eval_batch_size = args.eval_bs
    epochs = args.epochs
    name = args.name
    shuffle = args.shuffle

    num_choices = 4
    vars(args)["num_choices"] = num_choices
    assert eval_batch_size % num_choices == 0, "Eval batch size should be a multiple of num choices, which is 4 for CosmosQA"

    model = Model(
        name=name,
        num_choices=num_choices
    ).cuda()

    sep_token = model.tokenizer.sep_token

    optimizer = configure_optimizer(model, args)

    Path("results/csqahi/").mkdir(parents=True, exist_ok=True)
    lf_name = "results/csqahi/" + name.replace("/", "-") + ".txt"
    lf = open(lf_name, "a")
    lf.write(str(args) + "\n\n")
    lf.close()


    for e in range(epochs):
        train_loader, val_loader, test_loader = configure_dataloaders(
         sep_token, train_batch_size, eval_batch_size, shuffle, percent
        )

        train_loss, train_acc, _, train_f1 = train_or_eval_model(model, train_loader, optimizer, "Train")
        val_loss, val_acc, val_ins_acc, val_f1 = train_or_eval_model(model, val_loader, split="Val")
        test_loss, test_acc, test_ins_acc, test_f1 = train_or_eval_model(model, test_loader, split="Test")

        x = "Epoch {}: Loss: Train {}; Val {}; Test: {}".format(e + 1, train_loss, val_loss, test_loss)
        y1 = "Classification Acc: Train {}; Val {}; Test: {}".format(train_acc, val_acc, test_acc)
        y2 = "Classification Macro F1: Train {}; Val {}; Test: {}".format(train_f1, val_f1, test_f1)
        z = "Instance Acc: Val {}; Test: {}".format(val_ins_acc, test_ins_acc)

        print(x)
        print(y1)
        print(y2)
        print(z)
        print("---------")

        lf = open(lf_name, "a")
        lf.write(x + "\n" + y1 + "\n" + y2 + "\n" + z + "\n\n")
        lf.close()


    lf = open(lf_name, "a")
    lf.write("-" * 100 + "\n")
    lf.close()
