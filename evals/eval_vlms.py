import argparse
import json
import sqlite3
import os
import pandas as pd
from metrics import ConstraintDecoding, LLMJudge
from prompts import SetOfMarkPrompt, ZeroShotPrompt
from scenic_reasoning.utilities.common import project_root_dir
from sqlitedict import SqliteDict
from tqdm import tqdm
from vlms import GPT, Llama
from pathlib import Path
import pickle

DB_PATH = project_root_dir() / "data/databases_final"

bdd_path = project_root_dir() / "data/bdd_val_filtered"



def iterate_sqlite_db(db_path, my_vlm, my_metric, my_prompt):
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

    correctness = []
    idx = 0
    for table in dataframes:
        for index, row in tqdm(
            dataframes[table].iterrows(), total=len(dataframes[table])
        ):
            output_path = output_dir / f"{idx}.txt"
            if os.path.exists(output_path):
                print(f"Skipping {output_path}")
                continue

            d = row.to_dict()
            pkl_path, v = d["key"], json.loads(d["value"])
            pkl_path = str(bdd_path / pkl_path)

            with open(pkl_path, "rb") as f:
                image_data = pickle.load(f)

            image_path = image_data["name"]
            image_path = str(project_root_dir() / f"data/bdd100k/images/100k/val/{image_path}")
            qa_list = v["qa_list"]
            if not qa_list or qa_list == "Question not applicable":
                print("Empty question, skipping...")
                continue

            questions = [p[0] for p in qa_list]
            questions = ", ".join([item for i, item in enumerate(questions)])
            answers = [p[1] for p in qa_list]
            answers = ", ".join([item for i, item in enumerate(answers)])

            try:
                preds, prompt = my_vlm.generate_answer(image_path, questions, my_prompt)
            except Exception as e:
                print(e)
                continue
            
            correct = my_metric.evaluate(preds, answers)
            correctness.append(correct)

            import pdb
            pdb.set_trace()
            
            with open(str(output_path), "w") as log_file:
                log_file.write(f"Image Path: \n{image_path}\n")
                log_file.write(f"Questions: \n{questions}\n")
                log_file.write(f"Answers: \n{answers}\n")
                log_file.write(f"Prompt: \n{prompt}\n")
                log_file.write(f"Preds: \n{preds}\n")
                log_file.write(f"Correctness: \n{correct}\n")
                log_file.write("\n")
            
            idx += 1
            
        
    return sum(correctness) / len(correctness), q_count
            


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
        choices=["GPT", "Qwen", "Llama"],
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
        choices=["SetOfMarkPrompt", "ZeroShotPrompt"],
        help="Prompt to use for generating questions.",
    )

    args = parser.parse_args()

    db_path = str(DB_PATH / args.db_name)
    if args.vlm == "GPT":
        my_vlm = GPT()
    elif args.vlm == "Llama":
        my_vlm = Llama()

    my_metric = LLMJudge() if args.metric == "LLMJudge" else ConstraintDecoding()
    my_prompt = (
        SetOfMarkPrompt() if args.prompt == "SetOfMarkPrompt" else ZeroShotPrompt()
    )

    acc, q_count = iterate_sqlite_db(db_path, my_vlm, my_metric, my_prompt)
    print(f"Accuracy: {acc}")
    print(f"Total questions: {q_count}")

