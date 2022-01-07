#!/usr/bin/env python3

import json
import os

notebooks_dir = os.path.dirname(os.path.realpath(__file__))
notebooks = [x for x in os.listdir(notebooks_dir) if x.endswith(".ipynb")]

for notebook in notebooks:
    with open(os.path.join(notebooks_dir, notebook), "r") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            for output in cell.get("outputs", {}):
                if "image/png" in output.get("data", {}):
                    del output["data"]["image/png"]

    with open(os.path.join(notebooks_dir, notebook), "w") as f:
        json.dump(nb, f, indent=1)
