from transformers import EsmModel
import torch.nn as nn

# ========== 多任务模型 ==========
def unfreeze_layers(model, num_unfreeze_layers):
    num_layers = model.config.num_hidden_layers  # 33
    # print(model)
    unfreeze_start = num_layers - num_unfreeze_layers
    for name, param in model.named_parameters():
        if "encoder.layer" in name:
            layer_num = int(name.split('.')[2])  # 提取层的编号
            if layer_num >= unfreeze_start:
                param.requires_grad = True  # 解冻
            else:
                param.requires_grad = False  # 冻结
    print(f"Unfreezed last {num_unfreeze_layers} layers and embeddings.")


class MultiTaskESM2(nn.Module):

    def __init__(self, model_name, num_unfreeze_layers=5):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        print(num_unfreeze_layers, type(num_unfreeze_layers))
        unfreeze_layers(self.esm, num_unfreeze_layers)
        hidden_size = self.esm.config.hidden_size

        # 三个任务头
        self.host_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 三分类
        )

        self.virulence_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # 二分类
        )

        self.receptor_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # 二分类
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]

        return {
            "host": self.host_head(pooled),
            "virulence": self.virulence_head(pooled),
            "receptor": self.receptor_head(pooled)
        }
