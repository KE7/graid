import argparse
import ast
import io
import json
import os
import pickle
import random
import re
import sqlite3
from pathlib import Path

import pandas as pd
from PIL import Image
from scenic_reasoning.evaluator.metrics import ConstrainedDecoding, ExactMatch, LLMJudge
from scenic_reasoning.evaluator.prompts import (
    CoT,
    SetOfMarkPrompt,
    ZeroShotPrompt,
    ZeroShotPrompt_batch,
)
from scenic_reasoning.evaluator.vlms import (
    GPT,
    GPT_CD,
    GPT_CoT_CD,
    Gemini,
    Gemini_CD,
    Gemini_CoT_CD,
    Llama,
    Llama_CD,
    Llama_CoT_CD,
)
from scenic_reasoning.utilities.common import project_root_dir
from sqlitedict import SqliteDict
from torchvision import transforms
from tqdm import tqdm

random.seed(42)

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

    output_dir = db_path.split(".py")[0]
    output_dir = Path(output_dir)
    results_dir = output_dir / f"{my_vlm}_{my_prompt}_{my_metric}"
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
    sample_size = 100  # this is per table not across all tables
    # if str(my_metric) != "LLMJudge" and "Llama" in str(my_vlm):
    #     sample_size = 100
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
            sampled_df = filtered_df.sample(n=sample_size, random_state=42).reset_index(
                drop=True
            )
        else:
            print(
                f"Table '{table_name}' has only {len(filtered_df)} valid rows. Returning all."
            )
            sampled_df = filtered_df.copy()

        sampled_dataframes[table_name] = sampled_df

    all_correctness = []

    for table_idx, table in enumerate(sampled_dataframes):
        # Reset correctness list for each table
        correctness = []
        output_path = results_dir / f"{table_idx}.txt"
        os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
        
        # Variable to track if we need to sample more questions
        need_more_samples = False
        existing_scores = []
        
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                text = f.read()
                match = re.search(r"Correctness:\s*\n(.*?)\n", text)
                if match:
                    score = match.group(1)
                    score = ast.literal_eval(score)
                    if type(score) == list:
                        existing_scores = score
                        # Check if we have the right number of scores
                        if len(existing_scores) > sample_size:
                            # Only take up to sample_size
                            print(f"Found {len(existing_scores)} scores, using first {sample_size}")
                            existing_scores = existing_scores[:sample_size]
                            all_correctness.extend(existing_scores)
                        elif len(existing_scores) < sample_size:
                            # We need to sample more questions
                            print(f"Found only {len(existing_scores)} scores, need {sample_size - len(existing_scores)} more")
                            all_correctness.extend(existing_scores)
                            need_more_samples = True
                        else:
                            # We have exactly the right number
                            all_correctness.extend(existing_scores)
                    else:
                        # Single score case
                        existing_scores = [score]
                        all_correctness.append(score)
                        if sample_size > 1:
                            need_more_samples = True
                else:
                    print(f"No correctness score found in {output_path}, skipping...")
                    need_more_samples = True
            
            # If we don't need more samples, continue to the next table
            if not need_more_samples:
                print(f"Using existing scores from {output_path}")
                continue
            else:
                print(f"Need more samples for {output_path}, will sample {sample_size - len(existing_scores)} more")

        questions = []
        answers = []
        image, image_path = (
            None,
            None,
        )  # Initialize image_path to avoid 'possibly unbound' errors
        
        # If we already have some scores, only sample what we need more
        additional_samples_needed = max(0, sample_size - len(existing_scores))
        if additional_samples_needed > 0 and need_more_samples:
            print(f"Sampling {additional_samples_needed} more questions for {table}")
            # If we need fewer samples than available, select randomly
            if len(sampled_dataframes[table]) > additional_samples_needed:
                # Only get the additional samples we need
                sample_indices = random.sample(range(len(sampled_dataframes[table])), additional_samples_needed)
                rows_to_process = [sampled_dataframes[table].iloc[idx] for idx in sample_indices]
            else:
                # Use all available samples if we need more than we have
                rows_to_process = [sampled_dataframes[table].iloc[idx] for idx in range(len(sampled_dataframes[table]))]
        else:
            # Process all rows if we haven't cached any scores yet
            rows_to_process = [sampled_dataframes[table].iloc[idx] for idx in range(len(sampled_dataframes[table]))]
        
        for idx, row in tqdm(
            enumerate(rows_to_process), total=len(rows_to_process)
        ):

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
                image = transforms.ToTensor()(Image.open(io.BytesIO(image_path)))

            if isinstance(qa_list[0], list):
                random_qa = random.choice(qa_list)
                questions.append(random_qa[0])
                answers.append(random_qa[1])
                # questions += [item[0] for item in qa_list]
                # answers += [item[1] for item in qa_list]
            else:
                questions.append(qa_list[0])
                answers.append(qa_list[1])

            if len(questions) == 0:
                print(f"No questions found for image index {idx}, skipping...")
                continue

        preds = []
        prompt = ""

        if use_batch and image_path is not None:
            questions = ", ".join([item for i, item in enumerate(questions)])
            answers = ", ".join([item for i, item in enumerate(answers)])
            image_for_prompt, prompt = my_prompt.generate_prompt(image_path, questions)
            if image_path is not None:
                cache_key = f"{my_vlm}_{my_prompt}_{image_path}_{prompt}" + (
                    "_SoM" if "SetOfMarkPrompt" == str(my_prompt) else ""
                ) + "_batch"
                if cache_key in vlm_cache:
                    preds = vlm_cache[cache_key]
                else:
                    preds, prompt = my_vlm.generate_answer(
                        image_for_prompt, questions, my_prompt
                    )
                    vlm_cache[cache_key] = preds
            else:
                print("Warning: image_path is None, skipping batch processing")
                preds = []
            correct = my_metric.evaluate(preds, answers)
            correctness.append(correct)
            preds = preds
        else:
            for q, a in tqdm(zip(questions, answers), total=len(questions)):
                if len(q) < 5:  # check for "D" and "y"
                    raise ValueError(f"Question too short: {q}")

                # the cache key should be image_path + prompt
                image_for_prompt, prompt = my_prompt.generate_prompt(image, q)
                cache_key = f"{my_vlm}_{my_prompt}_{image_path}_{prompt}" + (
                    "_SoM" if "SetOfMarkPrompt" == str(my_prompt) else ""
                )
                if cache_key in vlm_cache:
                    pred = vlm_cache[cache_key]
                else:
                    pred, prompt = my_vlm.generate_answer(image_for_prompt, q, my_prompt)
                    vlm_cache[cache_key] = pred
                    vlm_cache.commit()
                correct = my_metric.evaluate(pred, a)

                preds.append(pred)
                correctness.append(correct)

        # Combine existing scores with new scores, if we loaded some from cache
        if need_more_samples and len(existing_scores) > 0:
            combined_correctness = existing_scores + correctness
            # Ensure we don't exceed sample_size
            if len(combined_correctness) > sample_size:
                combined_correctness = combined_correctness[:sample_size]
            all_correctness.append(combined_correctness)
            
            # Update the cached file with the combined scores
            with open(str(output_path), "w") as log_file:
                log_file.write(
                    f"Image Path: \n{image_path if image_path is not None else 'None'}\n"
                )
                log_file.write(f"Questions: \n{questions}\n")
                log_file.write(f"Answers: \n{answers}\n")
                log_file.write(f"Prompt: \n{prompt}\n")
                log_file.write(f"Preds: \n{preds}\n")
                log_file.write(f"Correctness: \n{combined_correctness}\n")
                log_file.write("\n")
        else:
            # Normal case without cached scores
            all_correctness.append(correctness)
            
            with open(str(output_path), "w") as log_file:
                log_file.write(
                    f"Image Path: \n{image_path if image_path is not None else 'None'}\n"
                )
                log_file.write(f"Questions: \n{questions}\n")
                log_file.write(f"Answers: \n{answers}\n")
                log_file.write(f"Prompt: \n{prompt}\n")
                log_file.write(f"Preds: \n{preds}\n")
                log_file.write(f"Correctness: \n{correctness}\n")
                log_file.write("\n")

    vlm_cache.close()
    conn.close()
    # flatten the list from a list of lists of floats to a list of floats
    all_correctness = [item if isinstance(item, (int, float)) else item for sublist in all_correctness for item in (sublist if isinstance(sublist, list) else [sublist])]
    return sum(all_correctness) / len(all_correctness)


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
        choices=[
            "GPT",
            "GPT_CD",
            "GPT_CoT_CD",
            "Gemini",
            "Gemini_CD",
            "Gemini_CoT_CD",
            "Llama",
            "Llama_CD",
            "Llama_CoT_CD",
        ],
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
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=7,
    )

    args = parser.parse_args()

    use_batch = False  # args.db_name = "bdd_val_0.2_/co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py.sqlite"; args.vlm="Llama_CD"; args.metric="ExactMatch"; args.prompt="ZeroShotPrompt"

    db_path = str(DB_PATH / args.db_name)
    if args.vlm == "GPT": # python evals/eval_vlms.py --db_name "bdd_val_0.2_/co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py.sqlite"  --vlm GPT_CoT_CD --metric ExactMatch --prompt CoT
        import pick
        choice_list = ["gpt-4.1-2025-04-14", "o4-mini-2025-04-16", "gpt-4o"]
        title = "Please select a GPT model:"
        option, index = pick.pick(choice_list, title)
        my_vlm = GPT(model_name=str(option))
        # use_batch = True
    elif args.vlm == "GPT_CD":
        import pick
        choice_list = ["gpt-4.1-2025-04-14", "o4-mini-2025-04-16", "gpt-4o"]
        title = "Please select a GPT model:"
        option, index = pick.pick(choice_list, title)   
        my_vlm = GPT_CD(model_name=str(option))
    elif args.vlm == "GPT_CoT_CD":
        import pick
        choice_list = ["gpt-4.1-2025-04-14", "o4-mini-2025-04-16", "gpt-4o"]
        title = "Please select a GPT model:"
        option, index = pick.pick(choice_list, title)   
        my_vlm = GPT_CoT_CD(model_name=str(option))
    elif args.vlm == "Llama":
        my_vlm = Llama()
        # use_batch = False
    elif args.vlm == "Llama_CD":
        my_vlm = Llama_CD()
    elif args.vlm == "Llama_CoT_CD":
        my_vlm = Llama_CoT_CD()
        # use_batch = False
    elif args.vlm == "Gemini":
        import pick
        choice_list = ["gemini-1.5-pro", "gemini-2.5-pro"]
        title = "Please select a Gemini model:"
        option, index = pick.pick(choice_list, title)
        
        my_vlm = Gemini(model_name=str(option), location=args.region)
        # use_batch = True
    elif args.vlm == "Gemini_CD":
        import pick
        choice_list = ["gemini-1.5-pro", "gemini-2.5-pro"]
        title = "Please select a Gemini model:"
        option, index = pick.pick(choice_list, title)
        
        my_vlm = Gemini_CD(model_name=str(option), location=args.region)
    elif args.vlm == "Gemini_CoT_CD":
        import pick
        choice_list = ["gemini-1.5-pro", "gemini-2.5-pro"]
        title = "Please select a Gemini model:"
        option, index = pick.pick(choice_list, title)
        
        my_vlm = Gemini_CoT_CD(model_name=str(option), location=args.region)
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
        if "GPT" in args.vlm:
            my_prompt = SetOfMarkPrompt(gpu=args.gpu_id)
        elif "Llama" in args.vlm:
            my_prompt = SetOfMarkPrompt(gpu=args.gpu_id)
        elif "Gemini" in args.vlm:
            my_prompt = SetOfMarkPrompt(gpu=args.gpu_id)
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
