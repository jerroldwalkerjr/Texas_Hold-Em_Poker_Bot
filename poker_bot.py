"""
poker_bot.py

Texas Hold'em poker bot for your Go/WebSocket engine.

Features:
- Card representation & hand evaluation (5-card + 7-card)
- Monte Carlo equity estimation
- Simple opponent modeling (VPIP / aggression)
- EV-based strategy (fold / call / raise / bet)
- WebSocket client that joins a table and plays automatically

Usage:
    pip install websockets

    python poker_bot.py <hero_id> <api_key> <table_name> <host:port>

Example:
    python poker_bot.py p1 dev table-1 localhost:8080
"""

import asyncio
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import List, Dict

import websockets

# ===============================
# Card representation & utilities
# ===============================

RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs, diamonds, hearts, spades
RANK_TO_INT = {r: i for i, r in enumerate(RANKS, start=2)}
INT_TO_RANK = {v: k for k, v in RANK_TO_INT.items()}


@dataclass(frozen=True)
class Card:
    rank: int  # 2–14
    suit: str  # 'c', 'd', 'h', 's'


def parse_card(cs: str) -> Card:
    """
    Parse a 2-char card string like 'Ah', 'Tc', '7d'.
    """
    if len(cs) != 2:
        raise ValueError(f"Bad card string: {cs!r}")
    rank_char = cs[0].upper()
    suit_char = cs[1].lower()
    if rank_char not in RANK_TO_INT or suit_char not in SUITS:
        raise ValueError(f"Bad card string: {cs!r}")
    return Card(RANK_TO_INT[rank_char], suit_char)


def card_str(card: Card) -> str:
    return f"{INT_TO_RANK[card.rank]}{card.suit}"


def full_deck() -> List[Card]:
    return [Card(RANK_TO_INT[r], s) for r in RANKS for s in SUITS]


def log_decision(state, equity, pot_odds, adj_equity, decision):
    """
    Print a readable decision block without altering game logic.
    """
    try:
        street = (state.street or "").upper()
        hole = ", ".join(card_str(c) for c in state.hole_cards) if state.hole_cards else "-"
        board = ", ".join(card_str(c) for c in state.board_cards) if state.board_cards else "-"
        eq = "N/A" if equity is None else f"{equity:.3f}"
        adj = "N/A" if adj_equity is None else f"{adj_equity:.3f}"
        po = "N/A" if pot_odds is None else f"{pot_odds:.3f}"
        block = [
            "---- DECISION ----",
            f"Street: {street}",
            f"Hole: {hole}",
            f"Board: {board}",
            f"Equity: {eq} | Adjusted: {adj} | Pot odds: {po}",
            f"Pot: {state.pot_size} | To call: {state.to_call}",
            f"Action: {decision}",
            "------------------",
        ]
        print("\n".join(block))
    except Exception:
        # Logging should never break gameplay.
        pass


# ==========================
# 5-card and 7-card evaluator
# ==========================

def evaluate_5cards(cards: List[Card]):
    """
    Return a rank tuple for a 5-card poker hand.
    Higher tuple => stronger hand.

    Category codes:
        8: Straight Flush
        7: Four of a Kind
        6: Full House
        5: Flush
        4: Straight
        3: Three of a Kind
        2: Two Pair
        1: One Pair
        0: High Card
    """
    assert len(cards) == 5
    ranks = sorted((c.rank for c in cards), reverse=True)
    counts = Counter(ranks)
    # list of (rank, count) sorted by (count, rank) desc
    count_rank = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    is_flush = len({c.suit for c in cards}) == 1

    # Straight detection (incl. wheel A-5)
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    high_straight = None
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            high_straight = unique_ranks[0]
        # A-5 straight: A,5,4,3,2 = 14,5,4,3,2
        elif unique_ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            high_straight = 5

    # Straight flush
    if is_flush and is_straight:
        return (8, high_straight)

    (r1, c1), (r2, c2), *rest = count_rank

    # Four of a Kind
    if c1 == 4:
        kicker = max(r for r in ranks if r != r1)
        return (7, r1, kicker)

    # Full House (3 + 2)
    if c1 == 3 and c2 == 2:
        return (6, r1, r2)

    # Flush
    if is_flush:
        return (5, *sorted(ranks, reverse=True))

    # Straight
    if is_straight:
        return (4, high_straight)

    # Three of a Kind
    if c1 == 3:
        kickers = sorted([r for r in ranks if r != r1], reverse=True)
        return (3, r1, *kickers)

    # Two Pair
    if c1 == 2 and c2 == 2:
        pair1, pair2 = sorted([r1, r2], reverse=True)
        kicker = max(r for r in ranks if r != pair1 and r != pair2)
        return (2, pair1, pair2, kicker)

    # One Pair
    if c1 == 2:
        pair = r1
        kickers = sorted([r for r in ranks if r != pair], reverse=True)
        return (1, pair, *kickers)

    # High Card
    return (0, *sorted(ranks, reverse=True))


def evaluate_7cards(cards: List[Card]):
    """
    Return the best 5-card hand rank out of 7 cards.
    """
    assert len(cards) == 7
    best = None
    for combo in combinations(cards, 5):
        score = evaluate_5cards(list(combo))
        if best is None or score > best:
            best = score
    return best


# ==========================
# Monte Carlo equity estimator
# ==========================

def estimate_equity(
    hole_cards: List[Card],
    board_cards: List[Card],
    num_opponents: int,
    num_samples: int = 1000,
) -> float:
    """
    Monte Carlo estimate of hero's win probability.

    Returns:
        equity in [0, 1] (ties count as half a win)
    """
    if num_opponents <= 0:
        return 1.0

    used = set(hole_cards + board_cards)
    base_deck = [c for c in full_deck() if c not in used]

    wins = 0
    ties = 0

    needed_board = 5 - len(board_cards)

    for _ in range(num_samples):
        deck = base_deck[:]
        random.shuffle(deck)

        cards_needed = needed_board + 2 * num_opponents
        sample = deck[:cards_needed]

        sim_board = board_cards + sample[:needed_board]
        opp_holes = [
            sample[needed_board + 2 * i : needed_board + 2 * (i + 1)]
            for i in range(num_opponents)
        ]

        hero_score = evaluate_7cards(hole_cards + sim_board)
        opp_scores = [evaluate_7cards(opp + sim_board) for opp in opp_holes]

        better = sum(1 for s in opp_scores if s > hero_score)
        equal = sum(1 for s in opp_scores if s == hero_score)

        if better == 0 and equal == 0:
            wins += 1
        elif better == 0 and equal > 0:
            ties += 1

    total = num_samples
    return (wins + 0.5 * ties) / total


# ===================
# Opponent Modeling
# ===================

@dataclass
class OpponentStats:
    hands_played: int = 0
    vpip: int = 0              # voluntarily put money in pot preflop
    preflop_raises: int = 0
    postflop_raises: int = 0
    calls: int = 0
    folds: int = 0

    def vpip_rate(self) -> float:
        return self.vpip / self.hands_played if self.hands_played > 0 else 0.0

    def pfr_rate(self) -> float:
        return self.preflop_raises / self.hands_played if self.hands_played > 0 else 0.0

    def aggression_factor(self) -> float:
        return self.postflop_raises / self.calls if self.calls > 0 else 0.0


class OpponentModel:
    def __init__(self):
        self.stats: Dict[str, OpponentStats] = defaultdict(OpponentStats)

    def record_hand_start(self, players: List[str]):
        for pid in players:
            self.stats[pid].hands_played += 1

    def record_action(self, player_id: str, street: str, action: str):
        """
        action: 'fold', 'call', 'raise', 'check', 'bet'
        street: 'preflop', 'flop', 'turn', 'river'
        """
        s = self.stats[player_id]
        street = street.lower()

        if street == "preflop":
            if action in ("call", "raise"):
                s.vpip += 1
            if action == "raise":
                s.preflop_raises += 1
        else:
            if action in ("raise", "bet"):
                s.postflop_raises += 1

        if action == "fold":
            s.folds += 1
        elif action == "call":
            s.calls += 1

    def classify_player(self, player_id: str) -> str:
        """
        Very rough labels: 'tight', 'loose', 'aggro', 'passive', 'unknown'
        """
        s = self.stats[player_id]
        if s.hands_played < 10:
            return "unknown"

        vpip = s.vpip_rate()
        pfr = s.pfr_rate()
        af = s.aggression_factor()

        if vpip < 0.15:
            return "tight"
        if vpip > 0.35:
            return "loose"

        # medium VPIP → differentiate by aggression
        if af > 2.0 or pfr > 0.25:
            return "aggro"
        return "passive"


# =====================
# Game state & strategy
# =====================

@dataclass
class GameState:
    hero_id: str
    hole_cards: List[Card]
    board_cards: List[Card]
    pot_size: float
    to_call: float
    min_raise: float
    hero_stack: float
    num_active_opponents: int
    street: str                 # 'preflop', 'flop', 'turn', 'river'
    opponent_ids: List[str]


class Strategy:
    def __init__(self, opp_model: OpponentModel):
        self.opp_model = opp_model

    @staticmethod
    def compute_pot_odds(state: GameState) -> float:
        """
        Pot odds = cost_to_call / (pot + cost_to_call)
        """
        if state.to_call <= 0:
            return 0.0
        return state.to_call / (state.pot_size + state.to_call)

    def choose_action(self, state: GameState) -> Dict:
        """
        Returns:
            {"action": "fold" | "call" | "raise" | "bet" | "check", "amount": optional}
        """
        def add_meta(decision: Dict) -> Dict:
            decision = dict(decision)
            decision["meta"] = {
                "equity": equity,
                "pot_odds": pot_odds,
                "adj_equity": adj_equity,
            }
            return decision

        # Estimate equity via Monte Carlo
        equity = estimate_equity(
            hole_cards=state.hole_cards,
            board_cards=state.board_cards,
            num_opponents=state.num_active_opponents,
            num_samples=800,
        )

        pot_odds = self.compute_pot_odds(state)

        # Opponent-based slight adjustment
        tight_count = sum(
            1 for pid in state.opponent_ids
            if self.opp_model.classify_player(pid) == "tight"
        )
        loose_count = sum(
            1 for pid in state.opponent_ids
            if self.opp_model.classify_player(pid) == "loose"
        )

        adj_equity = equity
        if tight_count > loose_count:
            adj_equity += 0.05  # table is tight → bluff/value more
        adj_equity = max(0.0, min(1.0, adj_equity))

        # Preflop special rules
        if state.street == "preflop":
            return add_meta(self.preflop_decision(state, adj_equity))

        # Postflop logic
        margin = 0.05
        if state.to_call > 0:
            if adj_equity + margin < pot_odds:
                return add_meta({"action": "fold"})

            # Call region
            if adj_equity < pot_odds + 0.15:
                return add_meta({"action": "call"})

            # Strong → raise/value
            raise_size = min(state.hero_stack, state.pot_size * 0.75 + state.to_call)
            if raise_size <= state.to_call:
                return add_meta({"action": "call"})
            return add_meta({"action": "raise", "amount": raise_size})
        else:
            # No bet to us (we can check or bet)
            if adj_equity < 0.35:
                return add_meta({"action": "check"})
            elif adj_equity < 0.55:
                # semi-bluff
                if random.random() < 0.4:
                    bet_size = min(state.hero_stack, state.pot_size * 0.5)
                    return add_meta({"action": "bet", "amount": bet_size})
                return add_meta({"action": "check"})
            else:
                # strong value bet
                bet_size = min(state.hero_stack, state.pot_size * 0.75)
                return add_meta({"action": "bet", "amount": bet_size})

    def preflop_decision(self, state: GameState, adj_equity: float) -> Dict:
        """
        Simple preflop strategy based on:
        - pocket pairs
        - broadway cards
        - suited / connectedness
        """
        c1, c2 = state.hole_cards
        ranks_sorted = sorted([c1.rank, c2.rank], reverse=True)
        same_suit = c1.suit == c2.suit
        high_card = ranks_sorted[0]
        low_card = ranks_sorted[1]

        is_pair = c1.rank == c2.rank
        is_premium_pair = is_pair and c1.rank >= RANK_TO_INT["T"]  # TT+
        is_mid_pair = is_pair and RANK_TO_INT["6"] <= c1.rank <= RANK_TO_INT["9"]

        is_broadway = high_card >= RANK_TO_INT["T"]
        gap = high_card - low_card

        if is_premium_pair or (is_broadway and same_suit and gap <= 3):
            # Premium hand
            if state.to_call == 0:
                raise_size = min(state.hero_stack,
                                 max(state.min_raise, state.pot_size * 0.5 + 1))
                return {"action": "raise", "amount": raise_size}
            else:
                raise_size = min(state.hero_stack, state.to_call * 3)
                return {"action": "raise", "amount": raise_size}

        if is_mid_pair or (is_broadway and gap <= 4):
            # Playable
            if state.to_call == 0:
                if random.random() < 0.8:
                    raise_size = min(state.hero_stack,
                                     max(state.min_raise, state.pot_size * 0.5 + 1))
                    return {"action": "raise", "amount": raise_size}
                return {"action": "check"}
            else:
                pot_odds = self.compute_pot_odds(state)
                if adj_equity >= pot_odds - 0.05:
                    return {"action": "call"}
                return {"action": "fold"}

        # Junk
        if state.to_call > 0:
            return {"action": "fold"}
        # Sometimes bluff-raise if folded to us
        if random.random() < 0.15:
            raise_size = min(state.hero_stack,
                             max(state.min_raise, state.pot_size * 0.4 + 1))
            return {"action": "raise", "amount": raise_size}
        return {"action": "check"}


# =======================
# WebSocket Poker Bot
# =======================

class PokerBot:
    def __init__(self, hero_id: str, api_key: str, table: str, host: str):
        self.hero_id = hero_id
        self.api_key = api_key
        self.table = table
        self.host = host  # e.g. "localhost:8080"
        self.uri = f"ws://{self.host}/ws?apiKey={self.api_key}&table={self.table}&player={self.hero_id}"

        self.opp_model = OpponentModel()
        self.strategy = Strategy(self.opp_model)

        # Keep around between states
        self.hole_cards: List[Card] = []
        self.current_players: List[str] = []

    async def run(self):
        async with websockets.connect(self.uri) as ws:
            # Join the table as per README
            await ws.send(json.dumps({"type": "join"}))
            print(f"[INFO] Joined as {self.hero_id} on {self.table} at {self.host}")

            while True:
                raw = await ws.recv()
                data = json.loads(raw)
                msg_type = data.get("type")

                if msg_type == "state":
                    await self.handle_state(ws, data)
                elif msg_type == "error":
                    print("[ENGINE ERROR]:", data.get("message"))
                else:
                    # Optionally log unknown messages
                    # print("[DEBUG] Unknown message:", data)
                    pass

    async def handle_state(self, ws, data: Dict):
        """
        IMPORTANT: This function maps the engine's JSON into our GameState.

        The README only shows:
            { "type": "state", "hand": 2, "phase": "TURN", "pot": 540 }

        In practice, the engine **must** be sending more, e.g.:
            - which player's turn it is
            - players and their stacks/bets
            - hero's hole cards
            - community cards
            - amount to call

        So:

        1) Run the engine and log `data` like:
               print(json.dumps(data, indent=2))
           to see the real structure.

        2) Then adjust the code in the marked TODO section below.
        """

        # Debug (compact summary so logs stay short)
        state_payload = data.get("state", {})
        table_payload = state_payload.get("table", {})
        players_payload = table_payload.get("players", [])
        turn_idx = state_payload.get("toActIdx")
        turn_id = None
        if isinstance(turn_idx, int) and 0 <= turn_idx < len(players_payload):
            turn_id = players_payload[turn_idx].get("id")
        phase_summary = (state_payload.get("phase") or table_payload.get("phase") or "").upper()
        print(
            f"[STATE] hand={state_payload.get('hand')} phase={phase_summary} "
            f"toActIdx={turn_idx} toActID={turn_id} pot={state_payload.get('pot')} "
            f"hero={self.hero_id} players={[p.get('id') for p in players_payload]}",
            flush=True,
        )

        phase_raw = data.get("phase", "").lower()  # "PREFLOP" → "preflop"
        street = phase_raw  # "preflop", "flop", "turn", "river", "showdown"? etc.

        # ==== TODO: BEGIN mapping engine JSON → our internal format ====

        # The following field names are assumptions.
        # You MUST check against actual messages from your engine and adjust.

        # Players array: list of dicts
        # Expected example:
        # "players": [
        #   { "id": "p1", "stack": 950, "bet": 50, "inHand": true, "cards": ["Ah","Kd"] },
        #   { "id": "p2", "stack": 1200, "bet": 100, "inHand": true }
        # ]
        players = data.get("players", [])
        if not players:
            # If the engine doesn't send players here, adjust accordingly.
            return

        # Which player's turn is it?
        # Expected field: "turn": "p1"
        turn_player = data.get("turn")

        # Pot size
        pot = float(data.get("pot", 0))

        # Amount hero must call
        # Expected field: "callAmount": 50
        to_call = float(data.get("callAmount", 0))

        # Community cards
        # Expected field: "community": ["Ah","9d","3s","Td"]
        community_raw = data.get("community", [])
        board_cards = [parse_card(c) for c in community_raw]

        # Find hero entry
        hero_entry = None
        for p in players:
            if p.get("id") == self.hero_id:
                hero_entry = p
                break
        if hero_entry is None:
            # Not seated / between hands
            return

        hero_stack = float(hero_entry.get("stack", 0))
        hero_bet = float(hero_entry.get("bet", 0))

        # Hero hole cards: we try several possible keys
        hero_cards_raw = hero_entry.get("cards") or hero_entry.get("hole") or hero_entry.get("hand") or []
        if len(hero_cards_raw) == 2:
            self.hole_cards = [parse_card(c) for c in hero_cards_raw]

        if len(self.hole_cards) != 2:
            # No hole cards yet (maybe between hands)
            return

        # Determine min_raise:
        # Typical logic: min raise = current highest bet * 2 - your bet
        max_bet = max(float(p.get("bet", 0)) for p in players)
        min_raise = max(max_bet * 2 - hero_bet, max_bet - hero_bet)

        # Active opponents still in the hand
        opponent_ids = [
            p.get("id")
            for p in players
            if p.get("id") != self.hero_id and p.get("inHand", True)
        ]

        num_active_opponents = len(opponent_ids)

        # ==== TODO: END mapping engine JSON → our internal format ====

        # Only act if it's actually our turn
        if turn_player != self.hero_id:
            return

        # Build GameState
        state = GameState(
            hero_id=self.hero_id,
            hole_cards=self.hole_cards,
            board_cards=board_cards,
            pot_size=pot,
            to_call=to_call,
            min_raise=min_raise,
            hero_stack=hero_stack,
            num_active_opponents=num_active_opponents,
            street=street,
            opponent_ids=opponent_ids,
        )

        decision = self.strategy.choose_action(state)

        print("=== MY TURN ===")
        print("Street:", street.upper())
        print("Hole:", [card_str(c) for c in self.hole_cards])
        print("Board:", [card_str(c) for c in board_cards])
        print("Pot:", pot, "To call:", to_call, "Stack:", hero_stack)
        print("Decision:", decision)

        meta = decision.get("meta", {}) if isinstance(decision, dict) else {}
        log_decision(
            state=state,
            equity=meta.get("equity"),
            pot_odds=meta.get("pot_odds"),
            adj_equity=meta.get("adj_equity"),
            decision=decision,
        )

        # Map internal decision → engine action message
        action_type = decision["action"]
        if action_type in ("check", "call", "fold"):
            msg = {"type": "action", "action": action_type}
        elif action_type in ("bet", "raise"):
            amount = float(decision.get("amount", 0))
            msg = {"type": "action", "action": "raise", "amount": round(amount, 2)}
        else:
            # Fallback safety
            msg = {"type": "action", "action": "fold"}

        await ws.send(json.dumps(msg))


# =================================
# Entry point
# =================================

def main():
    if len(sys.argv) != 5:
        print("Usage: python poker_bot.py <hero_id> <api_key> <table_name> <host:port>")
        print("Example: python poker_bot.py p1 dev table-1 localhost:8080")
        sys.exit(1)

    hero_id = sys.argv[1]
    api_key = sys.argv[2]
    table = sys.argv[3]
    host = sys.argv[4]

    bot = PokerBot(hero_id=hero_id, api_key=api_key, table=table, host=host)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
