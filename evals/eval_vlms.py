import argparse
import ast
import io
import json
import os
import pickle
import re
import sqlite3
from pathlib import Path
import pandas as pd
from scenic_reasoning.evaluator.metrics import ConstrainedDecoding, LLMJudge, ExactMatch
from PIL import Image
from scenic_reasoning.evaluator.prompts import CoT, SetOfMarkPrompt, ZeroShotPrompt, ZeroShotPrompt_batch
from scenic_reasoning.utilities.common import project_root_dir
from sqlitedict import SqliteDict
from torchvision import transforms
from tqdm import tqdm
from scenic_reasoning.evaluator.vlms import GPT, Gemini, Llama, Llama_CD, Llama_CoT, Llama_CoT_CD
import random

DB_PATH = project_root_dir() / "data/databases_ablations"

bdd_path = project_root_dir() / "data/bdd_val_filtered"
nu_path = project_root_dir() / "data/nuimages_val_filtered"
waymo_path = project_root_dir() / "data/waymo_validation_interesting"


def iterate_sqlite_db(db_path, my_vlm, my_metric, my_prompt, use_batch=False):
    if "bdd" in db_path:
        db_base_path = bdd_path
    elif "nuimage" in db_path:
        db_base_path = nu_path
    else:
        db_base_path = waymo_path

    conn = sqlite3.connect(db_path)
    l = db_path.split("/")
    db_path = "_".join([l[-2], l[-1]])

    output_dir = DB_PATH / f"{db_path}_{my_vlm}_{my_metric}_{my_prompt}"
    output_dir.mkdir(parents=True, exist_ok=True)

    vlm_cache_loc = output_dir / f"{my_vlm}_cache.db"
    vlm_cache = SqliteDict(
        str(vlm_cache_loc),
        tablename="vlm_cache",
        autocommit=True,
        encode=json.dumps,
        decode=json.loads,
    )

    # Get a list of all table names
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(tables_query, conn)["name"].tolist()

    dataframes = {}
    for table in tables:
        df = pd.read_sql(f"SELECT * FROM '{table}'", conn)
        dataframes[table] = df
    
    conn.close()


    sampled_dataframes = {}
    sample_size = 100
    print("Filtering rows...")

    for table_name, df in dataframes.items():
        filtered_rows = []
        for img_idx in tqdm(range(len(df))):
            row = df.iloc[img_idx]
            d = row.to_dict()

            pkl_path, v = d["key"], json.loads(d["value"])
            qa_list = v.get("qa_list", None)

            
            if not qa_list or qa_list == "Question not applicable":
                continue
            
            if isinstance(qa_list[0], list):
                qa_list = [random.choice(qa_list)]

            filtered_rows.append(row)

        filtered_df = pd.DataFrame(filtered_rows).reset_index(drop=True)
        if len(filtered_df) >= sample_size:
            sampled_df = filtered_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            print(f"Table '{table_name}' has only {len(filtered_df)} valid rows. Returning all.")
            sampled_df = filtered_df.copy()
            
        sampled_dataframes[table_name] = sampled_df

    correctness = []
    idx = 0

    for table in sampled_dataframes:
        output_path = output_dir / f"{img_idx}.txt"
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                text = f.read()
                match = re.search(r"Correctness:\s*\n(.*?)\n", text)
                score = match.group(1)
                score = ast.literal_eval(score)
                correctness.append(score)
            print(f"Skipping {output_path}")
            continue

        questions = []
        answers = []
        for img_idx, row in tqdm(sampled_dataframes[table].iterrows(), total=len(sampled_dataframes[table])):
            row = sampled_dataframes[table].iloc[img_idx]

            d = row.to_dict()
            pkl_path, v = d["key"], json.loads(d["value"])
            pkl_path = str(db_base_path / pkl_path)

            qa_list = v["qa_list"]
            if not qa_list or qa_list == "Question not applicable":
                print("Empty question, skipping...")
                continue

            with open(pkl_path, "rb") as f:
                image_data = pickle.load(f)

            if "bdd" in db_path:
                image_path = image_data["name"]
                image_path = str(
                    project_root_dir() / f"data/bdd100k/images/100k/val/{image_path}"
                )
            elif "nuimage" in db_path:
                image_path = image_data["filename"]
                image_path = str(
                    project_root_dir()
                    / f"/home/eecs/liheng/scenic-reasoning/data/nuimages/all/{image_path}"
                )
            else:
                image_path = transforms.ToTensor()(
                    Image.open(io.BytesIO(image_data["image"]))
                )


            if isinstance(qa_list[0], list):
                questions += [item[0] for item in qa_list]
                answers += [item[1] for item in qa_list]
            else:
                questions.append(qa_list[0])
                answers.append(qa_list[1])


        if not questions:
            print(f"No questions found for image index {img_idx}, skipping...")
            continue

        preds = []


        if use_batch:
            questions = ", ".join([item for i, item in enumerate(questions)])
            answers = ", ".join([item for i, item in enumerate(answers)])
            _, cache_key = my_prompt.generate_prompt(image_path, questions)
            if cache_key in vlm_cache:
                preds = vlm_cache[cache_key]
            else:
                preds, prompt = my_vlm.generate_answer(
                    image_path, questions, my_prompt
                )
                vlm_cache[cache_key] = preds
            correctness.append(my_metric.evaluate(preds, answers))
            preds = preds
        else:
            for q, a in tqdm(zip(questions, answers), total=len(questions)):
                if len(q) < 5:  #check for "D" and "y"
                    raise ValueError(f"Question too short: {q}")

                # the prompt is the cache key
                _, cache_key = my_prompt.generate_prompt(image_path, q)
                if cache_key in vlm_cache:
                    pred = vlm_cache[cache_key]
                else:
                    pred, prompt = my_vlm.generate_answer(image_path, q, my_prompt)
                    vlm_cache[cache_key] = pred
                correct = my_metric.evaluate(pred, a)

                preds.append(pred)
                correctness.append(correct)

        with open(str(output_path), "w") as log_file:
            log_file.write(f"Image Path: \n{image_path}\n")
            log_file.write(f"Questions: \n{questions}\n")
            log_file.write(f"Answers: \n{answers}\n")
            log_file.write(f"Prompt: \n{prompt}\n")
            log_file.write(f"Preds: \n{preds}\n")
            log_file.write(f"Correctness: \n{correctness}\n")
            log_file.write("\n")

    return sum(correctness) / len(correctness)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs using a SQLite database."
    )
    parser.add_argument(
        "--db_name",
        type=str,
        help="Path to the SQLite database.",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="Llama",
        choices=["GPT", "Gemini", "Llama", "Llama_CD"],
        help="VLM to use for generating answers.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="LLMJudge",
        choices=["ExactMatch", "LLMJudge", "ConstrainedDecoding"],
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

    use_batch = True

    db_path = str(DB_PATH / args.db_name)
    if args.vlm == "GPT":
        my_vlm = GPT()
        # use_batch = True
    elif args.vlm == "Llama":
        my_vlm = Llama(region=args.region)
        # use_batch = False
    elif args.vlm == "Llama_CD":
        my_vlm = Llama_CD(region=args.region)
        # use_batch = False
    elif args.vlm == "Gemini":
        my_vlm = Gemini(location=args.region)
        # use_batch = True

    if args.metric == "LLMJudge":
        my_metric = LLMJudge()
    elif args.metric == "ExactMatch":
        my_metric = ExactMatch()
    else:
        if args.vlm == "GPT":
            my_metric = ConstrainedDecoding(gpu=1, use_batch=use_batch)
        elif args.vlm == "Llama":
            my_metric = ConstrainedDecoding(gpu=2, use_batch=use_batch)
        elif args.vlm == "Gemini":
            my_metric = ConstrainedDecoding(gpu=3, use_batch=use_batch)


    if args.prompt == "SetOfMarkPrompt":
        if args.vlm == "GPT":
            my_prompt = SetOfMarkPrompt(gpu=1)
        elif args.vlm == "Llama":
            my_prompt = SetOfMarkPrompt(gpu=5)
        elif args.vlm == "Gemini":
            my_prompt = SetOfMarkPrompt(gpu=2)
        
    elif args.prompt == "CoT":
        if use_batch:
            my_prompt = CoT_batch()
        else:
            my_prompt = CoT()
    elif args.prompt == "ZeroShotPrompt":
        if use_batch:
            my_prompt = ZeroShotPrompt_batch()
        else:
            my_prompt = ZeroShotPrompt()

    
    acc = iterate_sqlite_db(db_path, my_vlm, my_metric, my_prompt, use_batch=use_batch)
    print(f"Accuracy: {acc}")
