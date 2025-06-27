import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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


def load_data(file_path, task_type):
    """Load data and add label columns for all tasks"""
    df = pd.read_csv(file_path,
                     names=["filename", "label", "sequence"],
                     skiprows=[0])

    # Initialize all task labels to -1 (indicating missing)
    df['host_label'] = -1
    df['virulence_label'] = -1
    df['receptor_label'] = -1

    # Fill in the current task label
    label_map = label_maps[task_type]
    df[task_type + '_label'] = df['label'].map(label_map)
    return df


def merge_datasets():
    """Merge multiple datasets into one"""
    host_df = load_data("./data/host_hapb2.csv", "host")
    virulence_df = load_data("./data/virulent_hapb2.csv", "virulence")
    receptor_df = load_data("./data/receptor.csv", "receptor")

    # Group by sequence and take the maximum label for each task
    merged_df = pd.concat([host_df, virulence_df,
                           receptor_df]).groupby('sequence').agg({
                               'host_label':
                               lambda x: x.max(),
                               'virulence_label':
                               'max',
                               'receptor_label':
                               'max'
                           }).reset_index()
    return merged_df


class MultiTaskDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length=1350):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = self.tokenizer(row["sequence"],
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt")

        return {
            "input_ids":
            inputs["input_ids"].squeeze(0),
            "attention_mask":
            inputs["attention_mask"].squeeze(0),
            "host_label":
            torch.tensor(row["host_label"], dtype=torch.long),
            "virulence_label":
            torch.tensor(row["virulence_label"], dtype=torch.long),
            "receptor_label":
            torch.tensor(row["receptor_label"], dtype=torch.long)
        }


def print_label_distribution(df):
    """Print how many labels each sample has (label completeness)"""
    label_counts = [
        sum(1 for task in ['host_label', 'virulence_label', 'receptor_label']
            if row[task] != -1) for _, row in df.iterrows()
    ]
    distribution = pd.Series(label_counts).value_counts().sort_index()
    print("\nLabel completeness distribution (before partitioning):")
    print("Samples with 3 labels:", distribution.get(3, 0))
    print("Samples with 2 labels:", distribution.get(2, 0))
    print("Samples with 1 label:", distribution.get(1, 0))
    print("Total number of samples:", len(df))


def print_task_label_counts(df, dataset_name):
    """Print label counts for each task in the specified dataset"""
    print(f"\n{dataset_name} label distribution:")

    # Host task label distribution
    host_counts = df[
        df['host_label'] != -1]['host_label'].value_counts().sort_index()
    host_str = "Host task: " + " | ".join([
        f"Category {label}: {host_counts.get(label, 0)}"
        for label in [0, 1, 2]
    ])

    # Virulence task label distribution
    virulence_counts = df[df['virulence_label'] !=
                          -1]['virulence_label'].value_counts().sort_index()
    virulence_str = "Virulence task: " + " | ".join([
        f"Category {label}: {virulence_counts.get(label, 0)}"
        for label in [0, 1]
    ])

    # Receptor task label distribution
    receptor_counts = df[df['receptor_label'] !=
                         -1]['receptor_label'].value_counts().sort_index()
    receptor_str = "Receptor task: " + " | ".join([
        f"Category {label}: {receptor_counts.get(label, 0)}"
        for label in [0, 1]
    ])

    print(host_str)
    print(virulence_str)
    print(receptor_str)


def data_process():

    # Data preparation
    all_data = merge_datasets()
    print_label_distribution(all_data)

    # Add a stratification column based on host_label
    all_data['stratify_col'] = all_data['host_label'].apply(lambda x: 3
                                                            if x == -1 else x)

    # Check the distribution of stratification column
    print("\nStratification column distribution:")
    print(all_data['stratify_col'].value_counts())

    # Ensure each stratum has at least 2 samples
    min_samples = 2
    valid_strata = all_data['stratify_col'].value_counts()[
        all_data['stratify_col'].value_counts() >= min_samples].index
    filtered_data = all_data[all_data['stratify_col'].isin(valid_strata)]

    # Split the data
    train_df, val_df = train_test_split(filtered_data,
                                        test_size=0.2,
                                        stratify=filtered_data['stratify_col'],
                                        random_state=42)

    remaining_data = all_data[~all_data.index.isin(filtered_data.index)]
    train_df = pd.concat([train_df, remaining_data])

    print_task_label_counts(train_df, "train dataset")
    print_task_label_counts(val_df, "valid dataset")

    model_path = './model/esm2_fz28'
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    train_loader = DataLoader(MultiTaskDataset(train_df, tokenizer),
                              batch_size=6,
                              shuffle=True)
    val_loader = DataLoader(MultiTaskDataset(val_df, tokenizer), batch_size=6)

    return train_loader, val_loader
