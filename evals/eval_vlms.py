import argparse
import json
import sqlite3
import os
import pandas as pd
from metrics import ConstraintDecoding, LLMJudge
from prompts import SetOfMarkPrompt, ZeroShotPrompt, CoT
from scenic_reasoning.utilities.common import project_root_dir
from sqlitedict import SqliteDict
from tqdm import tqdm
from vlms import GPT, Llama, Gemini
from pathlib import Path
import pickle
import ast
import re

DB_PATH = project_root_dir() / "data/databases_final"

bdd_path = project_root_dir() / "data/bdd_val_filtered"
nu_path = project_root_dir() / "data/nuimages_val_filtered"
waymo_path = project_root_dir() / "data/waymo_validation_interesting"



def iterate_sqlite_db(db_path, my_vlm, my_metric, my_prompt):
    if 'bdd' in db_path:
        db_base_path = bdd_path
    elif 'nuimage' in db_path:
        db_base_path = nu_path
    else:
        db_base_path = waymo_path

    conn = sqlite3.connect(db_path)
    db_name = Path(db_path).stem
    output_dir = DB_PATH / f"{db_name}_{my_vlm}_{my_metric}_{my_prompt}"
    output_dir.mkdir(parents=True, exist_ok=True)

    
    # Get a list of all table names
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(tables_query, conn)["name"].tolist()

    dataframes = {}
    for table in tables:
        df = pd.read_sql(f"SELECT * FROM '{table}'", conn)
        dataframes[table] = df

    conn.close()

    num_images = dataframes['Question: Is the width of the {object_1} appear to be larger than the height? (threshold: 0.3)'].shape[0]

    correctness = []
    idx = 0
    for img_idx in tqdm(range(num_images)):
        output_path = output_dir / f"{img_idx}.txt"
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                text = f.read()
                match = re.search(r"Correctness:\s*\n(.*?)\n", text)
                score = match.group(1)
                score = ast.literal_eval(score)
                correctness.extend(score)
            print(f"Skipping {output_path}")
            continue
        
        questions = []
        answers = []
        for table in dataframes:
            row = dataframes[table].iloc[img_idx]

            d = row.to_dict()
            pkl_path, v = d["key"], json.loads(d["value"])
            pkl_path = str(db_base_path / pkl_path)

            qa_list = v["qa_list"]
            if not qa_list or qa_list == "Question not applicable":
                print("Empty question, skipping...")
                continue

            with open(pkl_path, "rb") as f:
                image_data = pickle.load(f)

            if 'bdd' in db_path:
                image_path = image_data["name"]
                image_path = str(project_root_dir() / f"data/bdd100k/images/100k/val/{image_path}")
            elif 'nuimage' in db_path:
                image_path = image_data["filename"]
                image_path = str(project_root_dir() / f"/home/eecs/liheng/scenic-reasoning/data/nuimages/all/{image_path}")
            else:
                image_path = image_data["image"]
            

            questions += [p[0] for p in qa_list]
            answers += [p[1] for p in qa_list]
        
        if not questions:
            continue

        questions = ", ".join([item for i, item in enumerate(questions)])
        answers = ", ".join([item for i, item in enumerate(answers)])

        preds, prompt = my_vlm.generate_answer(image_path, questions, my_prompt)

        correct = my_metric.evaluate(preds, answers)
        correctness.extend(correct)

        with open(str(output_path), "w") as log_file:
            log_file.write(f"Image Path: \n{image_path}\n")
            log_file.write(f"Questions: \n{questions}\n")
            log_file.write(f"Answers: \n{answers}\n")
            log_file.write(f"Prompt: \n{prompt}\n")
            log_file.write(f"Preds: \n{preds}\n")
            log_file.write(f"Correctness: \n{correct}\n")
            log_file.write("\n")
            
    return sum(correctness) / len(correctness)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs using a SQLite database."
    )
    parser.add_argument(
        "--db_name",
        type=str,
        default="bdd_val_rtdetr-x.sqlite",
        help="Path to the SQLite database.",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="Llama",
        choices=["GPT", "Gemini", "Llama"],
        help="VLM to use for generating answers.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="LLMJudge",
        choices=["LLMJudge", "ConstraintDecoding"],
        help="Metric to use for evaluating answers.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="ZeroShotPrompt",
        choices=["SetOfMarkPrompt", "ZeroShotPrompt", "CoT"],
        help="Prompt to use for generating questions.",
    )

    parser.add_argument(
        "--region",
        type=str,
        default="us-central1",
    )

    args = parser.parse_args()

    db_path = str(DB_PATH / args.db_name)
    if args.vlm == "GPT":
        my_vlm = GPT()
    elif args.vlm == "Llama":
        my_vlm = Llama(region=args.region)
    elif args.vlm == "Gemini":
        my_vlm = Gemini(location=args.region)

    if args.metric == "LLMJudge":
        my_metric = LLMJudge()
    else:
        my_metric = ConstraintDecoding()
        

    if args.prompt == "SetOfMarkPrompt":
        my_prompt = SetOfMarkPrompt()
    elif args.prompt == "CoT":
        my_prompt = CoT()
    else:
        my_prompt = ZeroShotPrompt()

    acc = iterate_sqlite_db(db_path, my_vlm, my_metric, my_prompt)
    print(f"Accuracy: {acc}")

