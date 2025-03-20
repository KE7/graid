from sqlitedict import SqliteDict
from scenic_reasoning.utilities.common import project_root_dir
import sqlite3
import pandas as pd
import json
from vlms import GPT, Qwen
from metrics import LLMJudge, ConstraintDecoding
from prompts import SetOfMarkPrompt, ZeroShotPrompt
import argparse


DB_PATH = project_root_dir() / "scenic_reasoning/src/scenic_reasoning/data/databases"

def iterate_sqlite_db(db_path, my_vlm, my_metric, my_prompt):
    conn = sqlite3.connect(db_path)

    # Get a list of all table names
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(tables_query, conn)["name"].tolist()

    dataframes = {}
    for table in tables:
        df = pd.read_sql(f"SELECT * FROM '{table}'", conn)
        dataframes[table] = df 

    conn.close()

    correctness = []
    for table in dataframes:
        for index, row in dataframes[table].iterrows():
            d = row.to_dict()
            image_path, v = d['key'], json.loads(d['value'])
            qa_list = v['qa_list']
            if not qa_list or qa_list == 'Question not applicable':
                continue

            questions = [p[0] for p in qa_list]
            questions = "\n".join([f"{i+1}. {item}" for i, item in enumerate(questions)])
            answers = [p[1] for p in qa_list]
            
            preds = my_vlm.generate_answer(image_path, questions, my_prompt)
            for pred, answer in zip(preds, answers):
                correct = my_metric.evaluate(pred, answer)
                correctness.append(correct)
    
    return sum(correctness) / len(correctness)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLMs using a SQLite database.")
    parser.add_argument("--db_name", type=str, default="bdd_val_yolov8n.sqlite", help="Path to the SQLite database.")
    parser.add_argument("--vlm", type=str, default="GPT", choices=["GPT", "Qwen"], help="VLM to use for generating answers.")
    parser.add_argument("--metric", type=str, default="LLMJudge", choices=["LLMJudge", "ConstraintDecoding"], help="Metric to use for evaluating answers.")
    parser.add_argument("--prompt", type=str, default="ZeroShotPrompt", choices=["SetOfMarkPrompt", "ZeroShotPrompt"], help="Prompt to use for generating questions.")

    args = parser.parse_args()

    db_path = str(DB_PATH / args.db_name)
    my_vlm = GPT() if args.vlm == "GPT" else Qwen()
    my_metric = LLMJudge() if args.metric == "LLMJudge" else ConstraintDecoding()
    my_prompt = SetOfMarkPrompt() if args.prompt == "SetOfMarkPrompt" else ZeroShotPrompt()
    
    iterate_sqlite_db(db_path, my_vlm, my_metric, my_prompt)