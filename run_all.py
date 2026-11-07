"""Regenerate every number, table and figure of the revised manuscript.

Usage (Windows, macOS or Linux):
    python run_all.py
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("reproduce.py",   "Original protocol: standalone ANFIS equals ARX"),
    ("reproduce2.py",  "Original protocol: hybrid ANFIS"),
    ("experiments.py", "All models x 3 scaling schemes x 5 held-out events -> results/experiments.json"),
    ("extra.py",       "Optimiser sensitivity, LOEO summary, LOEO without the 1998 event"),
    ("tables.py",      "LaTeX tables -> results/tab_*.tex"),
    ("figures.py",     "Figures 3-6 -> figures/"),
]

def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    log_path = os.path.join(HERE, "results", "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        for script, desc in STEPS:
            print(f"[{script}] {desc}", flush=True)
            log.write(f"\n===== {script}: {desc} =====\n"); log.flush()
            r = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE,
                               stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                sys.exit(f"{script} failed; see {log_path}")
    print(f"Done. Console output of every step: {log_path}")

if __name__ == "__main__":
    main()
