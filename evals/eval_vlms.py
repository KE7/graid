from sqlitedict import SqliteDict
from scenic_reasoning.utilities.common import project_root_dir
import sqlite3
import pandas as pd
import json
from vlms import GPT, Qwen, Llama
from metrics import LLMJudge, ConstraintDecoding
from prompts import SetOfMarkPrompt, ZeroShotPrompt
import argparse
from tqdm import tqdm


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
    q_count = 0
    for table in dataframes:
        for index, row in tqdm(dataframes[table].iterrows(), total=len(dataframes[table])):
            if q_count == 1000:
                break
            d = row.to_dict()
            image_path, v = d['key'], json.loads(d['value'])
            qa_list = v['qa_list']
            if not qa_list or qa_list == 'Question not applicable':
                continue

            questions = [p[0] for p in qa_list]
            q_count += len(questions)
            questions = ", ".join([item for i, item in enumerate(questions)])
            answers = [p[1] for p in qa_list]
            answers = ", ".join([item for i, item in enumerate(answers)])
            try:
                preds = my_vlm.generate_answer(image_path, questions, my_prompt)
                if preds is None:
                    # Token has expired
                    return sum(correctness) / len(correctness)
                else:
                    correct = my_metric.evaluate(preds, answers)
                    correctness += correct
                print(q_count)
            except:
                return sum(correctness) / len(correctness), q_count

    
    return sum(correctness) / len(correctness), q_count
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLMs using a SQLite database.")
    parser.add_argument("--db_name", type=str, default="bdd_val_rtdetr-l.sqlite", help="Path to the SQLite database.")
    parser.add_argument("--vlm", type=str, default="Llama", choices=["GPT", "Qwen", "Llama"], help="VLM to use for generating answers.")
    parser.add_argument("--metric", type=str, default="LLMJudge", choices=["LLMJudge", "ConstraintDecoding"], help="Metric to use for evaluating answers.")
    parser.add_argument("--prompt", type=str, default="ZeroShotPrompt", choices=["SetOfMarkPrompt", "ZeroShotPrompt"], help="Prompt to use for generating questions.")

    args = parser.parse_args()

    db_path = str(DB_PATH / args.db_name)
    if args.vlm == "GPT":
        my_vlm = GPT()
    elif args.vlm == "Qwen":
        my_vlm = Qwen()
    else:
        my_vlm = Llama()

    my_metric = LLMJudge() if args.metric == "LLMJudge" else ConstraintDecoding()
    my_prompt = SetOfMarkPrompt() if args.prompt == "SetOfMarkPrompt" else ZeroShotPrompt()
    
    acc, q_count = iterate_sqlite_db(db_path, my_vlm, my_metric, my_prompt)
    print(f"Accuracy: {acc}")
    print(f"Total questions: {q_count}")
    result_file = f"results_{args.db_name}_{args.vlm}.txt"
    with open(result_file, "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Total questions: {q_count}\n")
        