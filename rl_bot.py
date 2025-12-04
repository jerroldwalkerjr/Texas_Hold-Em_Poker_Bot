import json
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

from pypokerengine.players import BasePokerPlayer

from poker_bot import Card, estimate_equity, parse_card


# -----------------------------
# Card helpers for the engine
# -----------------------------
def _convert_engine_card(raw: str) -> Card:
    raw = raw.strip()
    if len(raw) < 2:
        raise ValueError(f"Bad card string from engine: {raw!r}")
    suit = raw[0].lower()
    rank_part = raw[1:]
    rank_char = "T" if rank_part in ("T", "10") else rank_part.upper()
    return parse_card(f"{rank_char}{suit}")


def _active_opponents(round_state: dict, hero_uuid: str) -> List[str]:
    seats = round_state.get("seats", [])
    return [
        seat.get("uuid")
        for seat in seats
        if seat.get("uuid") != hero_uuid and seat.get("state") not in ("folded", "allin")
    ]


# -----------------------------
# RL Poker Bot
# -----------------------------
class RLPokerBot(BasePokerPlayer):
    """
    Q-learning bot with exploration decay, simplified state, reward shaping,
    and bootstrapped updates.
    """

    ACTIONS = ("fold", "call", "raise")

    def __init__(
        self,
        qtable_path: str = "qtable.json",
        epsilon: float = 0.7,
        alpha: float = 0.1,
        gamma: float = 0.95,
        training_enabled: bool = True,
    ):
        super().__init__()
        self.qtable_path = qtable_path
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.training_enabled = training_enabled

        self.qtable: Dict[str, Dict[str, float]] = {}
        self.trajectory: List[Tuple[str, str]] = []
        self.hand_start_stack: float = 0.0
        self.intermediate_reward: float = 0.0

        self.log_callback: Optional[Callable[[int, float, float, float, int], None]] = None
        self.total_reward: float = 0.0
        self.hands_played: int = 0
        self.hands_won: int = 0
        self.last_reward: float = 0.0

        self._load_qtable()

    # ------------------
    # Public controls
    # ------------------
    def set_log_callback(self, callback: Callable[[int, float, float, float, int], None]):
        self.log_callback = callback

    def set_epsilon(self, value: float):
        self.epsilon = max(0.0, min(1.0, value))

    def set_training(self, enabled: bool):
        self.training_enabled = enabled

    def decay_epsilon(self):
        """Decay exploration after a training phase."""
        self.epsilon = max(0.05, self.epsilon * 0.85)

    # ------------------
    # PyPokerEngine hooks
    # ------------------
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.trajectory = []
        self.intermediate_reward = 0.0
        hero = next((s for s in seats if s.get("uuid") == self.uuid), None)
        self.hand_start_stack = float(hero.get("stack", 0) if hero else 0)

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self._build_state(hole_card, round_state.get("community_card", []), round_state)
        chosen_action = self._select_action(state, valid_actions)

        # Reward shaping: folding preflop penalty
        if chosen_action == "fold" and (round_state.get("street", "preflop") or "preflop").lower() == "preflop":
            self.intermediate_reward -= 0.1

        self.trajectory.append((state, chosen_action))

        call_amt = next(a for a in valid_actions if a["action"] == "call")["amount"]

        if chosen_action == "fold":
            return "fold", 0

        if chosen_action == "call":
            check_action = next((a for a in valid_actions if a["action"] == "check"), None)
            if check_action:
                return "check", 0
            return "call", int(call_amt)

        raise_action = next((a for a in valid_actions if a["action"] == "raise"), None)
        if raise_action:
            min_raise = float(raise_action.get("amount", {}).get("min", 0) or 0)
            max_raise = float(raise_action.get("amount", {}).get("max", 0) or min_raise)
            amount = min_raise if min_raise > 0 else max_raise
            amount = amount if amount > 0 else call_amt
            return "raise", int(amount)

        # Fallback
        return "call", int(call_amt)

    def receive_street_start_message(self, street, round_state):
        street = (street or "").lower()
        if street == "flop":
            self.intermediate_reward += 0.1
        elif street == "turn":
            self.intermediate_reward += 0.2
        elif street == "river":
            self.intermediate_reward += 0.3

    def receive_round_result_message(self, winners, hand_info, round_state):
        hero_seat = next((s for s in round_state.get("seats", []) if s.get("uuid") == self.uuid), None)
        end_stack = float(hero_seat.get("stack", self.hand_start_stack) if hero_seat else self.hand_start_stack)
        stack_delta = end_stack - self.hand_start_stack

        # Bonus for winning
        if any(w.get("uuid") == self.uuid for w in winners or []):
            self.intermediate_reward += 1.0

        reward = self.intermediate_reward + stack_delta
        self.last_reward = reward
        self.total_reward += reward
        self.hands_played += 1
        if reward > 0:
            self.hands_won += 1
        win_rate = self.hands_won / self.hands_played if self.hands_played else 0.0

        if self.training_enabled:
            self._update_qtable(reward)
            self._save_qtable()

        if self.log_callback:
            self.log_callback(self.hands_played, reward, self.total_reward, win_rate, len(self.qtable))

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_game_result_message(self, winners, hand_info, round_state):
        pass

    # ------------------
    # RL internals
    # ------------------
    def _build_state(self, hole_card_raw: List[str], community_raw: List[str], round_state: dict) -> str:
        street = (round_state.get("street", "preflop") or "preflop").lower()
        hero_cards = [_convert_engine_card(c) for c in hole_card_raw]
        board_cards = [_convert_engine_card(c) for c in community_raw]
        opponents = len(_active_opponents(round_state, self.uuid))
        equity = self._estimate_equity(hero_cards, board_cards, opponents)
        equity_bucket = self._bucket_equity(equity)
        return f"{street}_eq{equity_bucket}"

    def _estimate_equity(self, hero_cards: List[Card], board_cards: List[Card], opponents: int) -> float:
        try:
            return estimate_equity(hero_cards, board_cards, opponents, num_samples=300)
        except Exception:
            return 0.5

    @staticmethod
    def _bucket_equity(equity: float) -> int:
        return min(4, int(equity * 5))

    def _select_action(self, state: str, valid_actions: list) -> str:
        available = self._available_actions(valid_actions)
        if not available:
            return "call"

        if self.training_enabled and random.random() < self.epsilon:
            return random.choice(available)

        q_for_state = self.qtable.get(state, {})
        best_action = max(available, key=lambda a: q_for_state.get(a, 0.0))
        return best_action

    @staticmethod
    def _available_actions(valid_actions: list) -> List[str]:
        actions = []
        for a in valid_actions:
            if a["action"] == "fold":
                actions.append("fold")
            elif a["action"] in ("call", "check"):
                if "call" not in actions:
                    actions.append("call")
            elif a["action"] == "raise":
                actions.append("raise")
        return actions

    def _update_qtable(self, terminal_reward: float):
        """
        Bootstrapped update through the trajectory.
        Only the final transition receives the terminal reward; others get 0 reward but
        propagate value via next_state bootstrap.
        """
        if not self.trajectory:
            return

        for i in reversed(range(len(self.trajectory))):
            state, action = self.trajectory[i]
            next_state = self.trajectory[i + 1][0] if i + 1 < len(self.trajectory) else None
            reward = terminal_reward if i == len(self.trajectory) - 1 else 0.0

            q_state = self.qtable.setdefault(state, {})
            current = q_state.get(action, 0.0)
            next_max = max(self.qtable.get(next_state, {}).values()) if next_state in self.qtable else 0.0
            target = reward + self.gamma * next_max
            q_state[action] = current + self.alpha * (target - current)

    def _load_qtable(self):
        if os.path.exists(self.qtable_path):
            try:
                with open(self.qtable_path, "r", encoding="utf-8") as f:
                    self.qtable = json.load(f)
            except Exception:
                self.qtable = {}
        else:
            self.qtable = {}

    def _save_qtable(self):
        os.makedirs(os.path.dirname(self.qtable_path) or ".", exist_ok=True)
        with open(self.qtable_path, "w", encoding="utf-8") as f:
            json.dump(self.qtable, f, indent=2, sort_keys=True)


# -----------------------------
# Simple opponents for training
# -----------------------------
class PassiveOpponent(BasePokerPlayer):
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
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
        call_amt = next(a for a in valid_actions if a["action"] == "call")["amount"]
        check_action = next((a for a in valid_actions if a["action"] == "check"), None)
        if check_action:
            return "check", 0
        if call_amt <= (round_state.get("small_blind_amount", 0) or 10) * 4:
            return "call", int(call_amt)
        return "fold", 0


class AggressiveOpponent(BasePokerPlayer):
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
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
        raise_action = next((a for a in valid_actions if a["action"] == "raise"), None)
        call_amt = next(a for a in valid_actions if a["action"] == "call")["amount"]

        if raise_action and random.random() < 0.6:
            min_r = float(raise_action.get("amount", {}).get("min", call_amt) or call_amt)
            max_r = float(raise_action.get("amount", {}).get("max", min_r) or min_r)
            amt = min_r if max_r <= 0 else max(min_r, max_r * 0.4)
            return "raise", int(amt)

        if random.random() < 0.8:
            return "call", int(call_amt)
        return "fold", 0


class SimpleEVOpponent(BasePokerPlayer):
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
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
        hero_cards = [_convert_engine_card(c) for c in hole_card]
        board_cards = [_convert_engine_card(c) for c in round_state.get("community_card", [])]
        opponents = len(_active_opponents(round_state, self.uuid))
        equity = 0.5
        try:
            equity = estimate_equity(hero_cards, board_cards, opponents, num_samples=200)
        except Exception:
            pass

        call_amt = next(a for a in valid_actions if a["action"] == "call")["amount"]
        raise_action = next((a for a in valid_actions if a["action"] == "raise"), None)

        if equity > 0.7 and raise_action:
            min_r = float(raise_action.get("amount", {}).get("min", call_amt) or call_amt)
            return "raise", int(min_r)
        if equity > 0.45:
            check_action = next((a for a in valid_actions if a["action"] == "check"), None)
            if check_action:
                return "check", 0
            return "call", int(call_amt)
        return "fold", 0


def make_monte_carlo_mirror(name: str = "mc_bot") -> BasePokerPlayer:
    from pypokerengine_runner import StrategyPlayer

    player = StrategyPlayer(hero_id=name)
    return player
