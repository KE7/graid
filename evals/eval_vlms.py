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
from scenic_reasoning.evaluator.vlms import (
    GPT, 
    GPT_CD,
    GPT_CoT_CD,
    Gemini, 
    Gemini_CD,
    Gemini_CoT_CD,
    Llama,
    Llama_CD,
    Llama_CoT_CD
)
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
    sample_size = 100 # this is per table not across all tables
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

    for table_idx, table in enumerate(sampled_dataframes):
        output_path = output_dir / f"{table_idx}.txt"
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                text = f.read()
                match = re.search(r"Correctness:\s*\n(.*?)\n", text)
                if match:
                    score = match.group(1)
                    score = ast.literal_eval(score)
                    correctness.append(score)
                else:
                    print(f"No correctness score found in {output_path}, skipping...")
            print(f"Skipping {output_path}")
            continue

        questions = []
        answers = []
        image, image_path = None, None  # Initialize image_path to avoid 'possibly unbound' errors
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
                image = image_path
            elif "nuimage" in db_path:
                image_path = image_data["filename"]
                image_path = str(
                    project_root_dir()
                    / f"/home/eecs/liheng/scenic-reasoning/data/nuimages/all/{image_path}"
                )
                image = image_path
            else:
                image_path = image_data["image"]
                image = transforms.ToTensor()(
                    Image.open(io.BytesIO(image_path))
                )


            if isinstance(qa_list[0], list):
                questions += [item[0] for item in qa_list]
                answers += [item[1] for item in qa_list]
            else:
                questions.append(qa_list[0])
                answers.append(qa_list[1])


            if len(questions) == 0:
                print(f"No questions found for image index {img_idx}, skipping...")
                continue

        preds = []
        prompt = ""
        
        if use_batch and image_path is not None:
            questions = ", ".join([item for i, item in enumerate(questions)])
            answers = ", ".join([item for i, item in enumerate(answers)])
            _, prompt = my_prompt.generate_prompt(image_path, questions)
            if image_path is not None:
                cache_key = f"{image_path}_{prompt}"
                if cache_key in vlm_cache:
                    preds = vlm_cache[cache_key]
                else:
                    preds, prompt = my_vlm.generate_answer(
                        image_path, questions, my_prompt
                    )
                    vlm_cache[cache_key] = preds
            else:
                print("Warning: image_path is None, skipping batch processing")
                preds = []
            correctness.append(my_metric.evaluate(preds, answers))
            preds = preds
        else:
            for q, a in tqdm(zip(questions, answers), total=len(questions)):
                if len(q) < 5:  #check for "D" and "y"
                    raise ValueError(f"Question too short: {q}")

                # the cache key should be image_path + prompt
                _, prompt = my_prompt.generate_prompt(image, q)
                cache_key = f"{image_path}_{prompt}"
                if cache_key in vlm_cache:
                    pred = vlm_cache[cache_key]
                else:
                    pred, prompt = my_vlm.generate_answer(image, q, my_prompt)
                    vlm_cache[cache_key] = pred
                correct = my_metric.evaluate(pred, a)

                preds.append(pred)
                correctness.append(correct)

        with open(str(output_path), "w") as log_file:
            log_file.write(f"Image Path: \n{image_path if image_path is not None else 'None'}\n")
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
        choices=["GPT", "GPT_CD", "GPT_CoT_CD", "Gemini", "Gemini_CD", "Gemini_CoT_CD", "Llama", "Llama_CD", "Llama_CoT_CD"],
        help="VLM to use for generating answers.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="LLMJudge",
        choices=["ExactMatch", "LLMJudge"],
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

    use_batch = False # args.db_name = "bdd_val_0.2_/co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py.sqlite"; args.vlm="Gemini_CD"; args.metric="ExactMatch"; args.prompt="ZeroShotPrompt"

    db_path = str(DB_PATH / args.db_name)
    if args.vlm == "GPT":
        my_vlm = GPT()
        # use_batch = True
    elif args.vlm == "GPT_CD":
        my_vlm = GPT_CD()
    elif args.vlm == "GPT_CoT_CD":
        my_vlm = GPT_CoT_CD()
    elif args.vlm == "Llama":
        my_vlm = Llama(region=args.region)
        # use_batch = False
    elif args.vlm == "Llama_CD":
        my_vlm = Llama_CD(region=args.region)
    elif args.vlm == "Llama_CoT_CD":
        my_vlm = Llama_CoT_CD(region=args.region)
        # use_batch = False
    elif args.vlm == "Gemini":
        my_vlm = Gemini(location=args.region)
        # use_batch = True
    elif args.vlm == "Gemini_CD":
        my_vlm = Gemini_CD(location=args.region)
    elif args.vlm == "Gemini_CoT_CD":
        my_vlm = Gemini_CoT_CD(location=args.region)
        # use_batch = False
    else:
        raise ValueError(f"Unknown VLM: {args.vlm}")

    if args.metric == "LLMJudge":
        my_metric = LLMJudge()
    elif args.metric == "ExactMatch":
        my_metric = ExactMatch()
    else:
        raise ValueError(f"Unknown metric: {args.metric}")

    if args.prompt == "SetOfMarkPrompt":
        if args.vlm == "GPT":
            my_prompt = SetOfMarkPrompt(gpu=1)
        elif args.vlm == "Llama":
            my_prompt = SetOfMarkPrompt(gpu=5)
        elif args.vlm == "Gemini":
            my_prompt = SetOfMarkPrompt(gpu=2)
        else:
            raise ValueError(f"SetOfMarkPrompt not supported for VLM: {args.vlm}")
        
    elif args.prompt == "CoT":
        if use_batch:
            raise ValueError("CoT does not support batch processing.")
        else:
            my_prompt = CoT()
    elif args.prompt == "ZeroShotPrompt":
        if use_batch:
            my_prompt = ZeroShotPrompt_batch()
        else:
            my_prompt = ZeroShotPrompt(using_cd=("CD" in args.vlm))
    else:
        raise ValueError(f"Unknown prompt: {args.prompt}")
    
    acc = iterate_sqlite_db(db_path, my_vlm, my_metric, my_prompt, use_batch=use_batch)
    print(f"Accuracy: {acc}")
