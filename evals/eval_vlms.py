from sqlitedict import SqliteDict
from scenic_reasoning.utilities.common import project_root_dir
import sqlite3
import pandas as pd
import json
from VLM import GPT
from metrics import LLMJudge
from prompts import SetOfMarkPrompt


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

    for table in dataframes:
        for index, row in dataframes[table].iterrows():
            d = row.to_dict()
            image_path, v = d['key'], json.loads(d['value'])
            qa_list = v['qa_list']
            if not qa_list:
                continue

            for question, answer in qa_list:
                pred = my_vlm.generate_answer(image_path, question, my_prompt)
                correct = my_metric.evaluate(pred, answer)
    


if __name__ == "__main__":
    db_path = DB_PATH / 'bdd_val_yolov8n.sqlite'
    db_path = str(db_path)
    my_vlm = GPT()
    my_metric = LLMJudge()
    my_prompt = SetOfMarkPrompt()
    iterate_sqlite_db(db_path, my_vlm, my_metric, my_prompt)