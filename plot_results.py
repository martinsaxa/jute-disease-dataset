import os
import glob
import re
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

RUNS_DIR = "runs"

def parse_run_name(run_name):
    # Example: efficientnet_epochs25_lr0.001
    m = re.match(r"(\w+)_epochs(\d+)_lr([0-9.e-]+)", run_name)
    if m:
        model, epochs, lr = m.groups()
        return model, int(epochs), float(lr)
    return None, None, None

results = []

for run_dir in sorted(os.listdir(RUNS_DIR)):
    log_dir = os.path.join(RUNS_DIR, run_dir)
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        continue
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    try:
        acc_events = ea.Scalars("Accuracy/test")
        f1_events = ea.Scalars("F1/test")
        if acc_events and f1_events:
            test_acc = acc_events[-1].value
            test_f1 = f1_events[-1].value
            model, epochs, lr = parse_run_name(run_dir)
            label = f"{model}/{epochs}/{lr:g}"
            results.append({
                "label": label,
                "test_acc": test_acc,
                "test_f1": test_f1,
            })
    except KeyError:
        continue

df = pd.DataFrame(results)
if df.empty:
    print("No results found!")
    exit()

# Sort by label for consistent plotting
df = df.sort_values("label")

# --- Bar plot for accuracy ---
plt.figure(figsize=(max(10, len(df)*0.8), 6))
plt.bar(df["label"], df["test_acc"], color="skyblue")
plt.ylabel("Test Accuracy")
plt.xlabel("Run (model/epochs/learning_rate)")
plt.title("Test Accuracy for Each Run")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("bar_test_accuracy.png")
plt.show()

# --- Bar plot for F1 ---
plt.figure(figsize=(max(10, len(df)*0.8), 6))
plt.bar(df["label"], df["test_f1"], color="salmon")
plt.ylabel("Test F1 Score")
plt.xlabel("Run (model/epochs/learning_rate)")
plt.title("Test F1 Score for Each Run")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("bar_test_f1.png")
plt.show()