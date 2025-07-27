import os
import json
import random
import argparse
from collections import defaultdict


def filter_fields(item):
    accuracy = None
    if "evaluation" in item and isinstance(item["evaluation"], dict):
        accuracy = item["evaluation"].get("accuracy", None)
    
    if accuracy is None:
        raise ValueError("Item does not contain 'evaluation' with 'accuracy' field.")
    
    return {
        "idx": item.get("idx", None),
        "accuracy": accuracy,
        "session_id": item.get("session_id", None),
        "domain": item.get("domain", None),
        "query_category": item.get("query_category", None),
        "dynamism": item.get("dynamism", None)
    }


def spilt_json(input_file_dir: str, output_dir: str, train_rate=0.8):
    if not os.path.exists(input_file_dir):
        raise FileNotFoundError(f"input file does not exist: {input_file_dir}")
    
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file_dir, 'r', encoding='utf-8') as f:
        all_info_list = json.load(f)

    domain_groups = defaultdict(list)
    for item in all_info_list:
        domain = item.get("domain", "unknown")
        domain_groups[domain].append(item)
    
    print(f"Total items: {len(all_info_list)} with {len(domain_groups)} domains.")

    train_list = []
    val_list = []
    for domain, items in domain_groups.items():
        random.shuffle(items)
        n_total = len(items)
        n_train = int(train_rate * n_total)
        train_list.extend([filter_fields(x) for x in items[:n_train]])
        val_list.extend([filter_fields(x) for x in items[n_train:]])

    train_path = os.path.join(output_dir, "train.json")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_list, f, indent=2, ensure_ascii=False)

    val_path = os.path.join(output_dir, "val.json")
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_list, f, indent=2, ensure_ascii=False)

    summary_list = [filter_fields(item) for item in all_info_list]
    summary_path = os.path.join(output_dir, "summary.json")

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_list, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split JSON data into train and validation sets.")
    parser.add_argument("input_file", type=str, 
                        help="Path to the input JSON file.")
    parser.add_argument("--output_dir", type=str, default="split_data", 
                        help="Directory to save the output files.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility.")
    
    args = parser.parse_args()

    random.seed(args.seed)
    spilt_json(args.input_file, args.output_dir)
