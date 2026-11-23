import json
import shutil
import os

nb_src = "D:/ag-ai/llm-classification-finetuning/submission_notebook.ipynb"
nb_dst = "D:/ag-ai/llm-classification-finetuning/submissions/submission_notebook.ipynb"
shutil.copyfile(nb_src, nb_dst)

with open(nb_src, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        try:
            compile(source, f"cell_{i}", "exec")
            print(f"Cell {i} compiled OK ({len(source.splitlines())} lines)")
        except SyntaxError as e:
            print(f"Cell {i} SYNTAX ERROR: {e}")
            raise e

print("All code cells successfully verified and compiled without errors.")
print(f"Copied to {nb_dst}")
