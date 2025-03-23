import argparse
import json
import sqlite3

import pandas as pd
from scenic_reasoning.utilities.common import project_root_dir
from tqdm import tqdm

# db_name = "bdd_train_rtdetr-l.sqlite"
# DB_PATH = project_root_dir() / "scenic_reasoning/src/scenic_reasoning/data/databases"
# db_path = str(DB_PATH / db_name)
# conn = sqlite3.connect(db_path)
# cursor = conn.cursor()

# # Get a list of all table names
# tables_query = "SELECT name FROM sqlite_master WHERE type='table';"

# tables = pd.read_sql(tables_query, conn)["name"].tolist()

# q_count = {}
# total_count = 0
# for table in tables:
#     df = pd.read_sql(f"SELECT * FROM '{table}'", conn)
#     print(f"Counting {table}")
#     count = 0
#     for index, row in tqdm(df.iterrows(), total=len(df)):
#         d = row.to_dict()
#         image_path, v = d['key'], json.loads(d['value'])
#         qa_list = v['qa_list']

#         if not qa_list or qa_list == 'Question not applicable':
#             continue

#         questions = [p[0] for p in qa_list]
#         count += len(questions)
#         total_count += len(questions)
#     print(f"Total questions in {table}: {count}")
#     q_count[table] = count

# q_count["total"] = total_count
# # Save the q_count as a JSON file
# output_path = f"{db_name[:-7]}_q_counts.json"
# with open(output_path, 'w') as f:
#     json.dump(q_count, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Count questions in SQLite database tables."
    )
    parser.add_argument(
        "--db_name",
        type=str,
        required=True,
        help="The SQLite database name (e.g., bdd_train_rtdetr-l.sqlite)",
    )
    args = parser.parse_args()

    db_name = args.db_name
    DB_PATH = (
        project_root_dir() / "data/databases3"
    )
    db_path = DB_PATH / db_name

    if not db_path.exists():
        print(f"Database path does not exist: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get table names
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(tables_query, conn)["name"].tolist()

    q_count = {}
    total_count = 0

    for table in tables:
        df = pd.read_sql(f"SELECT * FROM '{table}'", conn)
        print(f"Counting {table}...")
        count = 0
        for index, row in tqdm(df.iterrows(), total=len(df)):
            d = row.to_dict()
            image_path, v = d["key"], json.loads(d["value"])
            qa_list = v["qa_list"]

            if not qa_list or qa_list == "Question not applicable":
                continue

            questions = [p[0] for p in qa_list]
            count += len(questions)
            total_count += len(questions)
        print(f"Total questions in {table}: {count}")
        q_count[table] = count

    q_count["total"] = total_count

    # Save JSON
    output_path = f"{db_name[:-7]}_q_counts.json"
    with open(output_path, "w") as f:
        json.dump(q_count, f, indent=4)

    print(f"Counts saved to {output_path}")


if __name__ == "__main__":
    main()
