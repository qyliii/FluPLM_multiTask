# Influenza Protein Language Model for Multi-Task Phenotype Prediction

## Overview
This repository provides a **multi-task learning framework** built upon ESM-2 to simultaneously predict:

- Host species origin (avian/human/human_origin_avian)
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
- Fine-tune the last N(defualt 5) layers of the ESM-2 model with three classification heads simultaneously
- Save the trained checkpoint to ./model/esm2_fz28_unfz5_3task_2seq.pth


### Running Predictions

After training, you can predict phenotypes for a single protein sequence via command line:
```bash
python predict.py  <AMINO_ACID_SEQUENCE>
```

You will get output similar to:

Example output:
```bash
{
  "host": {
    "pred_label": "human",
    "pred_index": 2,
    "probs": [...]
  },
  "virulence": {
    "pred_label": "Virulent",
    "pred_index": 1,
    "prob": 0.92
  },
  "receptor": {
    "pred_label": "α2-6",
    "pred_index": 1,
    "prob": 0.87
  }
}
```

#### Notes
- Ensure input sequences contain valid amino acid letters.

- The model currently supports the three phenotype tasks listed above.

- For customization or further training, modify the `train.py` and `model.py` scripts accordingly.
