from pypokerengine.api.game import setup_config, start_poker

from rl_bot import (
    AggressiveOpponent,
    PassiveOpponent,
    RLPokerBot,
    SimpleEVOpponent,
    make_monte_carlo_mirror,
)

EVAL_HANDS = 1000
STACK = 2000
SMALL_BLIND = 10

OPPONENTS = [
    ("aggressive", AggressiveOpponent()),
    ("passive", PassiveOpponent()),
    ("ev", SimpleEVOpponent()),
    ("montecarlo", make_monte_carlo_mirror("mc")),
]


def evaluate_against(label: str, opponent):
    rl_bot = RLPokerBot(qtable_path="qtable.json", epsilon=0.0, alpha=0.0, training_enabled=False)
    rl_bot.set_training(False)

    config = setup_config(
        max_round=EVAL_HANDS,
        initial_stack=STACK,
        small_blind_amount=SMALL_BLIND,
    )
    config.register_player(name="rl", algorithm=rl_bot)
    config.register_player(name="opp", algorithm=opponent)
    config.register_player(name="filler", algorithm=PassiveOpponent())

    print(f"=== Evaluating vs {label} for {EVAL_HANDS} hands ===")
    start_poker(config, verbose=0)

    win_rate = rl_bot.hands_won / rl_bot.hands_played if rl_bot.hands_played else 0.0
    print(f"Win rate: {win_rate:.3f}")
    print(f"Total reward: {rl_bot.total_reward:.2f}")

    # Reset counters to avoid cross-contamination across opponents
    rl_bot.hands_played = 0
    rl_bot.hands_won = 0
    rl_bot.total_reward = 0


def main():
    for label, opp in OPPONENTS:
        evaluate_against(label, opp)


if __name__ == "__main__":
    main()
