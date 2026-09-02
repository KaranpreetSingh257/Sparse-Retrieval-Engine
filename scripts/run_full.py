import subprocess, sys

cmd = [
    sys.executable, "-m", "harness.run_harness",
    "--corpus", "data/full/corpus.jsonl",
    "--queries", "data/full/queries_dev.tsv",
    "--qrels", "data/full/qrels_dev.txt",
    "--run-out", "runs/dev_run.trec",
    "--report-out", "runs/dev_report.json"
]
subprocess.run(cmd)
