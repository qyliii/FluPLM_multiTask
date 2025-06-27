import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import torch.nn.functional as F
from data import *
from model import *



def calculate_loss(outputs, batch):
    loss = 0
    # 宿主任务损失（三分类）
    host_mask = (batch["host_label"] != -1)
    if host_mask.any():
        loss += F.cross_entropy(outputs["host"][host_mask],
                                batch["host_label"][host_mask])

    # 毒力任务损失（二分类）
    virulence_mask = (batch["virulence_label"] != -1)
    if virulence_mask.any():
        logits = outputs["virulence"][virulence_mask].view(-1)
        targets = batch["virulence_label"][virulence_mask].float()
        loss += F.binary_cross_entropy_with_logits(logits, targets)

    # 受体任务损失（二分类）
    receptor_mask = (batch["receptor_label"] != -1)
    if receptor_mask.any():
        logits = outputs["receptor"][receptor_mask].view(-1)
        targets = batch["receptor_label"][receptor_mask].float()
        loss += F.binary_cross_entropy_with_logits(logits, targets)

    return loss



def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        optimizer.zero_grad()

        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device)
        }
        labels = {
            "host_label": batch["host_label"].to(device),
            "virulence_label": batch["virulence_label"].to(device),
            "receptor_label": batch["receptor_label"].to(device)
        }

        outputs = model(**inputs)
        loss = calculate_loss(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    metrics = {
        "host": {
            "true": [],
            "pred": [],
            "probs": []
        },
        "virulence": {
            "true": [],
            "pred": [],
            "probs": []
        },
        "receptor": {
            "true": [],
            "pred": [],
            "probs": []
        }
    }

    with torch.no_grad():
        for batch in loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            outputs = model(**inputs)

            # host
            host_mask = (batch["host_label"] != -1).numpy()
            if host_mask.any():
                host_logits = outputs["host"].cpu()
                host_probs = F.softmax(host_logits, dim=1)
                host_preds = host_probs.argmax(dim=1).numpy()
                metrics["host"]["true"].extend(
                    batch["host_label"].numpy()[host_mask])
                metrics["host"]["pred"].extend(host_preds[host_mask])
                metrics["host"]["probs"].extend(host_probs.numpy()[host_mask])

            # virulence
            virulence_mask = (batch["virulence_label"] != -1).numpy()
            if virulence_mask.any():
                virulence_logits = outputs["virulence"].cpu().view(-1)
                virulence_probs = torch.sigmoid(virulence_logits).numpy()
                virulence_preds = (virulence_probs > 0.5).astype(int)
                metrics["virulence"]["true"].extend(
                    batch["virulence_label"].numpy()[virulence_mask])
                metrics["virulence"]["pred"].extend(
                    virulence_preds[virulence_mask])
                metrics["virulence"]["probs"].extend(
                    virulence_probs[virulence_mask])

            # receptor
            receptor_mask = (batch["receptor_label"] != -1).numpy()
            if receptor_mask.any():
                receptor_logits = outputs["receptor"].cpu().view(-1)
                receptor_probs = torch.sigmoid(receptor_logits).numpy()
                receptor_preds = (receptor_probs > 0.5).astype(int)
                metrics["receptor"]["true"].extend(
                    batch["receptor_label"].numpy()[receptor_mask])
                metrics["receptor"]["pred"].extend(
                    receptor_preds[receptor_mask])
                metrics["receptor"]["probs"].extend(
                    receptor_probs[receptor_mask])

    results = {}
    for task in metrics:
        if len(metrics[task]["true"]) == 0:
            continue

        true = np.array(metrics[task]["true"])
        pred = np.array(metrics[task]["pred"])
        probs = np.array(metrics[task]["probs"])

        accuracy = accuracy_score(true, pred)

        if task == "host":
            f1 = f1_score(true, pred, average='macro')
            precision = precision_score(true, pred, average='macro')
            recall = recall_score(true, pred, average='macro')
        else:
            f1 = f1_score(true, pred, average='binary')
            precision = precision_score(true, pred, average='binary')
            recall = recall_score(true, pred, average='binary')

        mcc = matthews_corrcoef(true, pred)

        try:
            if task == "host":
                auc = roc_auc_score(true,
                                    probs,
                                    multi_class='ovr',
                                    average='macro')
            else:
                auc = roc_auc_score(true, probs)
        except ValueError:
            auc = 0.0

        results[task] = {
            "accuracy": accuracy,
            "f1": f1,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "mcc": mcc
        }

    return results


def main():

    # data loader
    train_loader, val_loader = data_process()

    # model init
    model_path = './model/esm2_fz28' # or can use esm2 origin model see 
    unfreeze_layer = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskESM2(model_path,
                          num_unfreeze_layers=unfreeze_layer).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    

    # training
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(5):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_results = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        for task in val_results:
            print(f"\n{task} Metrics:")
            for metric, value in val_results[task].items():
                print(f"{metric}: {value:.4f} ", end='')
        print("\n")

    # save model
    torch.save(
        model.state_dict(),
        "./model/esm2_fz28_unfz5_3task_2seq.pth"
    )
    print("model save done.")
