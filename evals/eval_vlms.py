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

DB_PATH = project_root_dir() / "data/databases_final"


def iterate_sqlite_db(db_path, my_vlm, my_metric, my_prompt):
    conn = sqlite3.connect(db_path)
    db_name = Path(db_path).stem
    output_dir = Path(f"{db_name}_{my_vlm}_{my_metric}_{my_prompt}")
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
            idx += 1
            output_path = output_dir / f"{idx}.txt"
            if os.path.exists(output_path):
                print(f"Skipping {output_path}")
                continue

            d = row.to_dict()
            image_path, v = d["key"], json.loads(d["value"])
            qa_list = v["qa_list"]
            if not qa_list or qa_list == "Question not applicable":
                continue

            questions = [p[0] for p in qa_list]
            questions = ", ".join([item for i, item in enumerate(questions)])
            answers = [p[1] for p in qa_list]
            answers = ", ".join([item for i, item in enumerate(answers)])

            try:
                preds = my_vlm.generate_answer(image_path, questions, my_prompt)
            except Exception as e:
                print(e)
                continue
            
            correct = my_metric.evaluate(preds, answers)
            correctness.append(correct)
            
            with open(output_path, "w") as log_file:
                log_file.write(f"Image Path: \n{image_path}\n")
                log_file.write(f"Questions: \n{questions}\n")
                log_file.write(f"Answers: \n{answers}\n")
                log_file.write(f"Preds: \n{preds}\n")
                log_file.write(f"Correctness: \n{correct}\n")
                log_file.write("\n")
            
        
    return sum(correctness) / len(correctness), q_count
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs using a SQLite database."
    )
    parser.add_argument(
        "--db_name",
        type=str,
        default="bdd_val_rtdetr-l.sqlite",
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
    result_file = f"results_{args.db_name}_{args.vlm}.txt"
    with open(result_file, "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Total questions: {q_count}\n")
