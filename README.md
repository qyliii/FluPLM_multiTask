# Influenza Protein Language Model for Multi-Task Phenotype Prediction

## Overview
This repository provides a **multi-task learning framework** built upon protein language model to simultaneously predict:

- Host species (avian/human/human_origin_avian)
- Virulence level (Avirulent / Virulent)
- Receptor binding preference (α2-3 / α2-6)

It is designed for **biological sequence understanding** and aids in **phenotypic prediction of influenza virus**.

---

## Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/qyliii/FluPLM_multiTask.git
cd FluPLM_multiTask
pip install -r requirements.txt
```

## Usage
### Training the Model
To train the model, simply run:
```bash
python train.py
```

This will:
- Load and preprocess the datasets
- Fine-tune the last N (defualt 5) layers of the pretrained model with three classification heads simultaneously
- Save the trained checkpoint to ./model/esm2_flu_3task.pth

You can use either the [ESM2](https://github.com/facebookresearch/esm/) pretrained model or the fine-tuned [ESM2_Flu](https://drive.google.com/drive/folders/1d1BpyrjqdI_1sA2HcgM5VDFZdFlN5Icc?usp=drive_link) model on influenza protein sequences as the pretrained model.


### Running Predictions

After training, you can predict phenotypes for a single protein sequence via command line:
```bash
python predict.py  <AMINO_ACID_SEQUENCE>
```

You will get output similar to:

Example output:
```bash
{
  "host": { "pred_label": "human", "pred_index": 2, "probs": 0.98 },
  "virulence": { "pred_label": "Virulent",  "pred_index": 1, "prob": 0.92 },
  "receptor": { "pred_label": "α2-6", "pred_index": 1, "prob": 0.87 }
}
```

#### Notes
- Ensure input sequences contain valid amino acid letters.

- The model currently supports the three phenotype tasks listed above.

- For customization or further training, modify the `train.py` and `model.py` scripts accordingly.
