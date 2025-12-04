"""
simple_ev_bot.py

Simplified equity-based bot for the WebSocket Texas Hold'em engine.
Strategy (with Monte Carlo equity using 200 samples):
- equity < 0.30: fold to any bet, otherwise check
- 0.30 <= equity < 0.55: call bets up to 50% of stack, else fold; check if free
- equity >= 0.55: raise 50% pot when facing a bet (or call if raise not possible); when free to act, bet 40% pot
"""

import asyncio
import json
import random
import sys
from collections import Counter
from itertools import combinations
from typing import Dict, List

import websockets

# Card helpers (aligned with poker_bot.py)
RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_TO_INT = {r: i for i, r in enumerate(RANKS, start=2)}
INT_TO_RANK = {v: k for k, v in RANK_TO_INT.items()}


class Card:
    def __init__(self, rank: int, suit: str):
        self.rank = rank
        self.suit = suit


def parse_card(cs: str) -> Card:
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


def evaluate_5cards(cards: List[Card]):
    ranks = sorted((c.rank for c in cards), reverse=True)
    counts = Counter(ranks)
    count_rank = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    is_flush = len({c.suit for c in cards}) == 1

    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    high_straight = None
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            high_straight = unique_ranks[0]
        elif unique_ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            high_straight = 5

    if is_flush and is_straight:
        return (8, high_straight)

    (r1, c1), (r2, c2), *rest = count_rank

    if c1 == 4:
        kicker = max(r for r in ranks if r != r1)
        return (7, r1, kicker)
    if c1 == 3 and c2 == 2:
        return (6, r1, r2)
    if is_flush:
        return (5, *sorted(ranks, reverse=True))
    if is_straight:
        return (4, high_straight)
    if c1 == 3:
        kickers = sorted([r for r in ranks if r != r1], reverse=True)
        return (3, r1, *kickers)
    if c1 == 2 and c2 == 2:
        pair1, pair2 = sorted([r1, r2], reverse=True)
        kicker = max(r for r in ranks if r != pair1 and r != pair2)
        return (2, pair1, pair2, kicker)
    if c1 == 2:
        pair = r1
        kickers = sorted([r for r in ranks if r != pair], reverse=True)
        return (1, pair, *kickers)
    return (0, *sorted(ranks, reverse=True))


def evaluate_7cards(cards: List[Card]):
    best = None
    for combo in combinations(cards, 5):
        score = evaluate_5cards(list(combo))
        if best is None or score > best:
            best = score
    return best


def estimate_equity(hole_cards: List[Card], board_cards: List[Card], num_opponents: int, num_samples: int = 200) -> float:
    """
    Monte Carlo win probability with 200 samples by default.
    """
    if num_opponents <= 0:
        return 1.0

    used = set(hole_cards + board_cards)
    deck = [c for c in full_deck() if c not in used]
    needed_board = 5 - len(board_cards)

    wins = 0
    ties = 0
    for _ in range(num_samples):
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
    return (wins + 0.5 * ties) / total if total > 0 else 0.0


class SimpleEVBot:
    def __init__(self, hero_id: str, api_key: str, table: str, host: str):
        self.hero_id = hero_id
        self.api_key = api_key
        self.table = table
        self.host = host
        self.uri = f"ws://{self.host}/ws?apiKey={self.api_key}&table={self.table}&player={self.hero_id}"
        self.hole_cards: List[Card] = []

    async def run(self):
        async with websockets.connect(self.uri) as ws:
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
                    pass

    async def handle_state(self, ws, data: Dict):
        """
        Expects engine fields similar to poker_bot.py: players, turn, pot, callAmount, community, phase.
        """
        players = data.get("players", [])
        if not players:
            return

        turn_player = data.get("turn")
        pot = float(data.get("pot", 0))
        to_call = float(data.get("callAmount", 0))
        community_raw = data.get("community", [])
        board_cards = [parse_card(c) for c in community_raw]
        phase = (data.get("phase") or "").lower()

        hero_entry = None
        for idx, p in enumerate(players):
            if p.get("id") == self.hero_id:
                hero_entry = p
                break
        if hero_entry is None:
            return

        hero_stack = float(hero_entry.get("stack", 0))
        hero_cards_raw = hero_entry.get("cards") or hero_entry.get("hole") or hero_entry.get("hand") or []
        if len(hero_cards_raw) == 2:
            self.hole_cards = [parse_card(c) for c in hero_cards_raw]
        if len(self.hole_cards) != 2:
            return

        # Only act on our turn
        if turn_player != self.hero_id:
            return

        # Count active opponents still in hand
        num_active_opponents = len(
            [
                p for p in players
                if p.get("id") != self.hero_id and p.get("inHand", True)
            ]
        )

        equity = estimate_equity(
            hole_cards=self.hole_cards,
            board_cards=board_cards,
            num_opponents=num_active_opponents,
            num_samples=200,
        )

        action, amount = self.decide_action(equity, pot, to_call, hero_stack)

        print("=== SIMPLE EV TURN ===")
        print("Phase:", phase.upper())
        print("Hole:", [card_str(c) for c in self.hole_cards])
        print("Board:", [card_str(c) for c in board_cards])
        print(f"Pot: {pot} To call: {to_call} Stack: {hero_stack} Opponents: {num_active_opponents}")
        print(f"Equity: {equity:.3f}")
        print("Decision:", action, f"Amount: {amount}" if amount is not None else "")

        if amount is None:
            msg = {"type": "player_action", "action": action}
        else:
            msg = {"type": "player_action", "action": action, "amount": round(amount, 2)}

        await ws.send(json.dumps(msg))

    def decide_action(self, equity: float, pot: float, to_call: float, hero_stack: float):
        """
        Return (action, amount or None) based on simplified equity thresholds.
        """
        # If facing a bet
        if to_call > 0:
            if equity < 0.30:
                return "fold", None

            if equity < 0.55:
                if to_call <= 0.5 * hero_stack:
                    return "call", None
                return "fold", None

            # equity >= 0.55: attempt raise 50% pot, else call
            raise_amount = to_call + pot * 0.5
            raise_amount = min(hero_stack, raise_amount)
            if raise_amount > to_call:
                return "raise", raise_amount
            return "call", None

        # No bet to us (we can check/bet)
        if equity >= 0.55:
            bet_amount = pot * 0.4
            if bet_amount <= 0:
                bet_amount = min(hero_stack, 1.0)  # fallback small bet if pot is zero
            bet_amount = min(hero_stack, bet_amount)
            return "raise", bet_amount  # engine uses "raise" for bets

        return "check", None


def main():
    if len(sys.argv) != 5:
        print("Usage: python simple_ev_bot.py <hero_id> <api_key> <table_name> <host:port>")
        print("Example: python simple_ev_bot.py p1 dev table-1 localhost:8080")
        sys.exit(1)

    hero_id = sys.argv[1]
    api_key = sys.argv[2]
    table = sys.argv[3]
    host = sys.argv[4]

    bot = SimpleEVBot(hero_id=hero_id, api_key=api_key, table=table, host=host)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
