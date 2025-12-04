import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LOG_PATH = "training_log.csv"
QTABLE_PATH = "qtable.json"
GRAPH_DIR = "graphs"


def _ensure_graph_dir():
    os.makedirs(GRAPH_DIR, exist_ok=True)


def plot_win_rate(df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="hand_index", y="win_rate", hue="phase", estimator="mean", errorbar=None)
    plt.title("Win rate vs hands")
    plt.xlabel("Hand index")
    plt.ylabel("Win rate")
    plt.ylim(0, 1)
    plt.legend(title="Phase")
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, "win_rate.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved win rate plot to {path}")


def plot_rewards(df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="hand_index", y="cumulative_reward", hue="phase", errorbar=None)
    plt.title("Cumulative reward during training")
    plt.xlabel("Hand index")
    plt.ylabel("Cumulative reward (chips)")
    plt.legend(title="Phase")
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, "reward_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved reward curve to {path}")


def plot_q_heatmaps():
    if not os.path.exists(QTABLE_PATH):
        print(f"No Q-table found at {QTABLE_PATH}; skipping heatmaps")
        return
    with open(QTABLE_PATH, "r", encoding="utf-8") as f:
        qtable = json.load(f)

    actions = ["fold", "call", "raise"]
    states = list(qtable.keys())
    if not states:
        print("Q-table empty; skipping heatmaps")
        return

    # Limit to first 40 states to keep plot readable.
    max_states = 40
    states = states[:max_states]
    data = [[qtable.get(state, {}).get(act, 0.0) for act in actions] for state in states]
    df = pd.DataFrame(data, columns=actions, index=states)

    plt.figure(figsize=(10, max(6, len(states) * 0.25)))
    sns.heatmap(df, annot=False, cmap="viridis")
    plt.title("Q-values by state/action")
    plt.xlabel("Action")
    plt.ylabel("State")
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, "q_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved Q-value heatmap to {path}")


def main():
    _ensure_graph_dir()
    if not os.path.exists(LOG_PATH):
        print(f"Training log {LOG_PATH} not found")
        return

    df = pd.read_csv(LOG_PATH)
    plot_win_rate(df)
    plot_rewards(df)
    plot_q_heatmaps()


if __name__ == "__main__":
    sns.set_theme(style="darkgrid")
    main()
