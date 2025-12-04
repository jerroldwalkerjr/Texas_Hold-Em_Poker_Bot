import csv
import os
from typing import Callable, Tuple

from pypokerengine.api.game import setup_config, start_poker

from rl_bot import (
    AggressiveOpponent,
    PassiveOpponent,
    RLPokerBot,
    SimpleEVOpponent,
    make_monte_carlo_mirror,
)

TRAIN_HANDS_PER_PHASE = 10000
PHASES: Tuple[Tuple[str, Callable], ...] = (
    ("aggressive", AggressiveOpponent),
    ("passive", PassiveOpponent),
    ("ev", SimpleEVOpponent),
    ("montecarlo", make_monte_carlo_mirror),
)
STACK = 2000
SMALL_BLIND = 10
LOG_PATH = "training_log.csv"
QTABLE_PATH = "qtable.json"


def _init_log(log_path: str):
    header = ["phase", "hand", "reward", "cumulative", "win_rate", "qtable_size"]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def train():
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    _init_log(LOG_PATH)

    rl_bot = RLPokerBot(qtable_path=QTABLE_PATH, epsilon=0.2, alpha=0.1, gamma=0.95)

    current_phase = ""

    def log_callback(hand_idx: int, reward: float, cumulative: float, win_rate: float, q_size: int):
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([current_phase, hand_idx, reward, cumulative, win_rate, q_size])

    rl_bot.set_log_callback(log_callback)

    for phase_name, opp_factory in PHASES:
        current_phase = phase_name
        print(f"Starting training block for {phase_name} ({TRAIN_HANDS_PER_PHASE} hands)...")
        phase_start = rl_bot.hands_played
        phase_hands = 0

        while phase_hands < TRAIN_HANDS_PER_PHASE:
            remaining = TRAIN_HANDS_PER_PHASE - phase_hands

            # Build table with exactly 3 players: rl, target opp, passive filler.
            config = setup_config(
                max_round=remaining,
                initial_stack=STACK,
                small_blind_amount=SMALL_BLIND,
            )
            config.register_player(name="rl", algorithm=rl_bot)
            if phase_name == "montecarlo":
                opp_player = opp_factory("mc")
            else:
                opp_player = opp_factory()
            config.register_player(name="opp", algorithm=opp_player)
            config.register_player(name="filler", algorithm=PassiveOpponent())

            before = rl_bot.hands_played
            start_poker(config, verbose=0)
            after = rl_bot.hands_played
            delta = after - before
            phase_hands = after - phase_start

            print(
                f"  Chunk complete: +{delta} hands, phase total {phase_hands}/{TRAIN_HANDS_PER_PHASE}, overall {after}"
            )

            # Safety valve to prevent infinite loops in case of engine issues.
            if delta <= 0:
                print("  Warning: no hands played in last chunk; breaking early to avoid stalling.")
                break

        print(f"Finished {phase_name} block. Hands played so far: {rl_bot.hands_played}")
        rl_bot.decay_epsilon()  # decay exploration after this phase

    total_hands = rl_bot.hands_played
    print("Training complete")
    print(f"Total hands trained: {total_hands}")
    print(f"Q-table path: {QTABLE_PATH}")
    print(f"Log path: {LOG_PATH}")


if __name__ == "__main__":
    train()
