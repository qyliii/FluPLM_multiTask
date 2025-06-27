import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import *
import sys


def load_model(model_path, checkpoint_path, unfreeze_layer, device):
    model = MultiTaskESM2(model_path, num_unfreeze_layers=unfreeze_layer)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def prepare_input(sequence, tokenizer, device, max_length=1350):
    encoded = tokenizer(sequence,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=max_length)

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


label_maps = {
    "host": {
        "avian": 0,
        "human_origin_avian": 1,
        "human": 2
    },
    "virulence": {
        "Avirulent": 0,
        "Virulent": 1
    },
    "receptor": {
        "α2-3": 0,
        "α2-6": 1
    }
}

# 反转label_maps，方便根据数字标签查文字
inv_label_maps = {
    task: {v: k
           for k, v in label_maps[task].items()}
    for task in label_maps
}


def predict_single_sequence(sequence, model, tokenizer, device):
    inputs = prepare_input(sequence, tokenizer, device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = {}

    # 宿主任务（三分类）
    host_probs = F.softmax(outputs["host"], dim=1).squeeze().cpu().numpy()
    host_pred = host_probs.argmax()
    results["host"] = {
        "pred_label": inv_label_maps["host"][host_pred],  # 转文字标签
        "pred_index": int(host_pred),
        "probs": host_probs.tolist()
    }

    # 毒力任务（二分类）
    virulence_logit = outputs["virulence"].view(-1).item()
    virulence_prob = torch.sigmoid(torch.tensor(virulence_logit)).item()
    virulence_pred = int(virulence_prob > 0.5)
    results["virulence"] = {
        "pred_label": inv_label_maps["virulence"][virulence_pred],
        "pred_index": virulence_pred,
        "prob": virulence_prob
    }

    # 受体任务（二分类）
    receptor_logit = outputs["receptor"].view(-1).item()
    receptor_prob = torch.sigmoid(torch.tensor(receptor_logit)).item()
    receptor_pred = int(receptor_prob > 0.5)
    results["receptor"] = {
        "pred_label": inv_label_maps["receptor"][receptor_pred],
        "pred_index": receptor_pred,
        "prob": receptor_prob
    }

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <INFLUENZA_VIRUS_AMINO_ACID_SEQUENCE>")
        sys.exit(1)

    sequence = sys.argv[1]

    model_path = "./model/esm2_fz28"
    checkpoint_path = "./model/esm2_fz28_unfz5_3task_2seq.pth"
    unfreeze_layer = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_model(model_path, checkpoint_path, unfreeze_layer, device)

    result = predict_single_sequence(sequence, model, tokenizer, device)

    print("\nPrediction result:")
    for task in result:
        print(f"{task}: {result[task]}")