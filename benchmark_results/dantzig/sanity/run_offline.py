"""Run this small MILP study using gurobipy's bundled size-limited license.

This changes license selection only in the current process and its children.
No user license file is edited, and the selected license needs no network.
"""
import os
from pathlib import Path
import runpy

import gurobipy


if __name__ == "__main__":
    bundled = Path(gurobipy.__file__).parent/".libs"/"gurobi.lic"
    if not bundled.is_file():
        raise SystemExit("No bundled Gurobi license found; use run_study.py with your normal license")
    os.environ["GRB_LICENSE_FILE"] = str(bundled)
    runpy.run_path(str(Path(__file__).with_name("run_study.py")), run_name="__main__")
