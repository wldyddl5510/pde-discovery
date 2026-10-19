"""Summarize the actually constructed oracle errors, without running a solver."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    records = [json.loads(path.read_text()) for path in (directory / "prepared").glob("*.json")]
    rows = []
    for name in manifest["pdes"]:
        values = [record for record in records if record["name"] == name]
        if not values:
            continue
        for mode in ("rate", "exact"):
            pilots = [record["oracle"][mode] for record in values]
            if any(p["energy"] > p["target_energy"] for p in pilots):
                raise ValueError(f"Oracle bound violated: {name}/{mode}")
            rows.append(dict(name=name, mode=mode, realizations=len(pilots),
                n_regression=pilots[0]["n_regression"], n_total=pilots[0]["n_total"],
                delta_z=pilots[0]["delta_z"], target_energy=pilots[0]["target_energy"],
                alpha_min=min(p["alpha"] for p in pilots), alpha_max=max(p["alpha"] for p in pilots),
                max_weighted_energy=max(p["energy"] for p in pilots),
                max_raw_energy=max(p["energy_unweighted"] for p in pilots),
                max_energy_over_target=max(p["energy"] / p["target_energy"] for p in pilots),
                max_energy_over_n_quarter=max(p["error_over_n_quarter"] for p in pilots)))
    payload = dict(prepared_realizations=len(records), expected_realizations=manifest["expected_realizations"],
                   actual_oracle_pilots=2*len(records), by_pde=rows)
    (directory / "pilot_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    if rows:
        with (directory / "pilot_summary.csv").open("w", newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Constructed oracle-pilot error budgets", "",
        f"Prepared {len(records)}/{manifest['expected_realizations']} paired observations; "
        f"{2*len(records)} oracle pilots. These values summarize the actual floating-point states.", "",
        "E is the squared vector error integrated with physical Delta z and the split time weights. "
        "The rate target is n_reg^(-1/2); E/n_reg^(-1/4) <= n_reg^(-1/4) tends to zero. "
        "The exact oracle has E=0. Raw Delta z sums are also recorded in CSV/JSON.", "",
        "| PDE | Oracle | N | n_reg | Delta z | Target E | Alpha range | Maximum E | Max E/target | Max E/n_reg^(-1/4) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['name']} | {row['mode']} | {row['realizations']} | {row['n_regression']} | "
            f"{row['delta_z']:.6g} | {row['target_energy']:.6g} | "
            f"{row['alpha_min']:.6g} - {row['alpha_max']:.6g} | {row['max_weighted_energy']:.6g} | "
            f"{row['max_energy_over_target']:.9g} | {row['max_energy_over_n_quarter']:.6g} |")
    (directory / "pilot_summary.md").write_text("\n".join(lines) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    result = summarize(parser.parse_args().directory.resolve())
    print(f"Summarized {result['prepared_realizations']}/{result['expected_realizations']} observations, "
          f"{result['actual_oracle_pilots']} oracle pilots.")
