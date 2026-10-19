"""Select the bundled offline license only for this process and run the study."""
import os
from pathlib import Path
import gurobipy

if __name__ == "__main__":
    bundled = Path(gurobipy.__file__).parent/".libs"/"gurobi.lic"
    if not bundled.is_file():
        raise SystemExit("Bundled license unavailable; run run_study.py with your normal license")
    os.environ["GRB_LICENSE_FILE"] = str(bundled)
    import run_study as study
    # Frozen study manifests serialize tuple job keys as JSON arrays.  Compare
    # their canonical JSON-safe form on restart without changing numerical
    # source files or weakening any source/data/cache hash checks.
    original_register = study.prior.register_manifest
    def register_json_manifest(path, manifest):
        return original_register(path, study.serializable(manifest))
    study.prior.register_manifest = register_json_manifest
    study.main()
