"""
pypokerengine_runner.py

Bridge to run poker_bot.Strategy inside PyPokerEngine for offline testing.
"""

import argparse
import random
from typing import List

from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer

from poker_bot import Card, GameState, OpponentModel, Strategy, parse_card


def _convert_engine_card(raw: str) -> Card:
    raw = raw.strip()
    if len(raw) < 2:
        raise ValueError(f"Bad card string from engine: {raw!r}")
    suit = raw[0].lower()
    rank_part = raw[1:]
    rank_char = "T" if rank_part in ("T", "10") else rank_part.upper()
    return parse_card(f"{rank_char}{suit}")


def _calc_pot(round_state: dict) -> float:
    pot_info = round_state.get("pot", {})
    main_raw = pot_info.get("main", 0) or 0
    # PyPokerEngine may send `main` as a dict like {"amount": 30, "eligibles": [...]}
    if isinstance(main_raw, dict):
        main_amount = main_raw.get("amount", 0) or 0
    else:
        main_amount = main_raw
    side_total = 0
    for side in pot_info.get("side", []):
        if isinstance(side, dict):
            side_total += side.get("amount", 0) or 0
        else:
            side_total += side or 0
    return float(main_amount + side_total)


def _active_opponents(round_state: dict, hero_uuid: str) -> List[str]:
    seats = round_state.get("seats", [])
    return [
        seat.get("uuid")
        for seat in seats
        if seat.get("uuid") != hero_uuid and seat.get("state") not in ("folded", "allin")
    ]


def _hero_position(seats: List[dict], hero_uuid: str) -> str:
    """
    Derive a coarse position bucket (early/middle/late) from seating order.
    """
    hero_idx = None
    for i, seat in enumerate(seats):
        if seat.get("uuid") == hero_uuid:
            hero_idx = i
            break
    if hero_idx is None or not seats:
        return "middle"

    n = len(seats)
    if hero_idx < n / 3:
        return "early"
    if hero_idx >= (2 * n) / 3:
        return "late"
    return "middle"


def _was_last_aggressor_preflop(round_state: dict, hero_uuid: str) -> bool:
    """
    Inspect action history to see who last raised/bet preflop.
    """
    histories = round_state.get("action_histories", {}) or {}
    pre = histories.get("preflop", []) or []
    for action in reversed(pre):
        act = (action.get("action") or "").upper()
        if act in ("RAISE", "BET", "BIGBLIND", "SMALLBLIND", "ANTE"):
            return action.get("uuid") == hero_uuid
    return False


class StrategyPlayer(BasePokerPlayer):
    def __init__(self, hero_id: str = "hero"):
        super().__init__()
        self.hero_id = hero_id
        self.opp_model = OpponentModel()
        self.strategy = Strategy(self.opp_model)

    def receive_game_start_message(self, game_info):
        # Nothing needed, but PyPokerEngine requires the hook.
        pass

    def declare_action(self, valid_actions, hole_card, round_state):
        to_call = next(a for a in valid_actions if a["action"] == "call")["amount"]
        raise_action = next((a for a in valid_actions if a["action"] == "raise"), None)
        min_raise = 0.0
        max_raise = 0.0
        if raise_action:
            amounts = raise_action.get("amount", {}) or {}
            min_raise = float(amounts.get("min", 0))
            max_raise = float(amounts.get("max", 0))

        board_cards = [_convert_engine_card(c) for c in round_state.get("community_card", [])]
        hero_cards = [_convert_engine_card(c) for c in hole_card]
        seats = round_state.get("seats", [])
        hero_seat = next((s for s in seats if s.get("uuid") == self.uuid), None)
        hero_stack = float(hero_seat.get("stack", 0) if hero_seat else 0)

        street = round_state.get("street", "preflop").lower()
        opponent_ids = _active_opponents(round_state, self.uuid)
        hero_position = _hero_position(seats, self.uuid)
        hero_was_last_agg_pre = _was_last_aggressor_preflop(round_state, self.uuid)

        gs = GameState(
            hero_id=self.hero_id,
            hole_cards=hero_cards,
            board_cards=board_cards,
            pot_size=_calc_pot(round_state),
            to_call=float(to_call),
            min_raise=float(min_raise),
            hero_stack=hero_stack,
            num_active_opponents=len(opponent_ids),
            street=street,
            opponent_ids=opponent_ids,
            hero_position=hero_position,
            hero_was_last_aggressor_preflop=hero_was_last_agg_pre,
        )

        decision = self.strategy.choose_action(gs)
        action = decision.get("action")
        amount = float(decision.get("amount", 0) or 0)

        if action == "fold":
            return "fold", 0
        if action in ("check", "call"):
            check_action = next((a for a in valid_actions if a["action"] == "check"), None)
            if check_action:
                return "check", 0
            return "call", to_call
        if action in ("raise", "bet") and raise_action:
            chosen = max(min_raise, amount)
            if max_raise > 0:
                chosen = min(chosen, max_raise)
            if chosen <= 0:
                chosen = min_raise or to_call
            return "raise", int(chosen)
        return "call", to_call

    def receive_round_start_message(self, round_count, hole_card, seats):
        player_ids = [s.get("uuid") for s in seats]
        self.opp_model.record_hand_start(player_ids)

    def receive_street_start_message(self, street, round_state):
        # No additional setup needed per street.
        pass

    def receive_game_update_message(self, action, round_state):
        # Could be used to track opponent tendencies; skipped for now.
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # Hook for post-round updates; not needed for basic sim.
        pass

    def receive_game_result_message(self, winners, hand_info, round_state):
        # Hook for final game results; not needed for basic sim.
        pass


class SimpleRandomPlayer(BasePokerPlayer):
    """
    Minimal random policy to use as table opponents.
    Chooses uniformly among valid actions, picking min raise when raising.
    """

    def receive_game_start_message(self, game_info):
        # No setup needed for the random bot.
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        # Required hook; no state needed for random bot.
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def receive_game_result_message(self, winners, hand_info, round_state):
        pass

    def declare_action(self, valid_actions, hole_card, round_state):
        action = random.choice(valid_actions)
        act_type = action["action"]
        if act_type == "raise":
            amount = action["amount"]["min"]
            return "raise", amount
        if act_type == "call":
            return "call", action["amount"]
        return "fold", 0


def run_simulation(
    hands: int = 50,
    opponents: int = 3,
    stack: int = 2000,
    small_blind: int = 10,
    verbose: int = 1,
):
    # PyPokerEngine uses the parameter name `small_blind_amount`
    config = setup_config(
        max_round=hands,
        initial_stack=stack,
        small_blind_amount=small_blind,
    )
    config.register_player(name="hero", algorithm=StrategyPlayer(hero_id="hero"))
    for i in range(opponents):
        config.register_player(name=f"random_{i + 1}", algorithm=SimpleRandomPlayer())
    result = start_poker(config, verbose=verbose)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run PyPokerEngine simulation with the poker_bot Strategy bot."
    )
    parser.add_argument("--hands", type=int, default=50, help="Number of hands to simulate")
    parser.add_argument(
        "--opponents", type=int, default=3, help="Random opponents to seat"
    )
    parser.add_argument(
        "--stack", type=int, default=2000, help="Initial stack per player"
    )
    parser.add_argument(
        "--small-blind", dest="small_blind", type=int, default=10, help="Small blind amount"
    )
    parser.add_argument(
        "--verbose", type=int, default=1, choices=[0, 1, 2], help="PyPokerEngine verbosity"
    )
    args = parser.parse_args()

    summary = run_simulation(
        hands=args.hands,
        opponents=args.opponents,
        stack=args.stack,
        small_blind=args.small_blind,
        verbose=args.verbose,
    )

    print("=== Simulation complete ===")
    players_data = {}
    raw_players = None

    if isinstance(summary, dict):
        raw_players = summary.get("players")
    elif isinstance(summary, list):
        raw_players = summary

    if isinstance(raw_players, dict):
        players_data = raw_players
    elif isinstance(raw_players, list):
        # Normalize list of player dicts to {name: stack}
        for entry in raw_players:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("uuid")
                stack = entry.get("stack")
                if name is not None and stack is not None:
                    players_data[name] = stack

    if players_data:
        for player, stack in players_data.items():
            print(f"{player}: {stack}")
    else:
        print(summary)


if __name__ == "__main__":
    main()
