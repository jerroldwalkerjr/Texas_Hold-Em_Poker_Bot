"""
maniac_bot.py

Ultra-aggressive "maniac" bot for the WebSocket Texas Hold'em engine.
Behavior:
- Preflop: 60% raise, 40% call. Only fold if facing an all-in >60% of stack.
- Postflop: facing bet -> 40% raise, 40% call, 20% fold. No bet -> bet 40-70% pot.

Usage:
    pip install websockets
    python maniac_bot.py <hero_id> <api_key> <table_name> <host:port>
"""

import asyncio
import json
import random
import sys
from typing import Dict, List

import websockets

# Card helpers (mirrors poker_bot.py for compatibility)
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


class ManiacBot:
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
                    pass  # ignore unknown messages

    async def handle_state(self, ws, data: Dict):
        """
        Reads the same assumed engine payload fields as poker_bot.py:
            players, turn, pot, callAmount, community, phase
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
        hero_idx = None
        for idx, p in enumerate(players):
            if p.get("id") == self.hero_id:
                hero_entry = p
                hero_idx = idx
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

        action, amount = self.decide_action(phase, pot, to_call, hero_stack)

        # Log for visibility each street
        print("=== MANIAC TURN ===")
        print("Phase:", phase.upper())
        print("Hole:", [card_str(c) for c in self.hole_cards])
        print("Board:", [card_str(c) for c in board_cards])
        print(f"Pot: {pot} To call: {to_call} Stack: {hero_stack}")
        print("Chosen:", action, "Amount:" if amount is not None else "", amount if amount is not None else "")

        if amount is None:
            msg = {"type": "player_action", "action": action}
        else:
            msg = {"type": "player_action", "action": action, "amount": round(amount, 2)}

        await ws.send(json.dumps(msg))

    def decide_action(self, phase: str, pot: float, to_call: float, hero_stack: float):
        """
        Returns (action, amount or None)
        """
        phase = (phase or "").lower()

        # Preflop logic
        if phase == "preflop":
            if to_call > 0 and to_call > 0.6 * hero_stack:
                # Exception: fold to huge all-in
                return "fold", None

            roll = random.random()
            if roll < 0.6:
                # Raise sizing uses same rule as postflop raise formula
                raise_size = min(hero_stack, max(to_call * 2, pot * 0.5))
                if raise_size <= to_call:  # ensure actual raise beyond call
                    raise_size = min(hero_stack, to_call)
                return "raise", raise_size
            else:
                return ("call", None) if to_call > 0 else ("check", None)

        # Postflop logic
        if to_call > 0:
            roll = random.random()
            raise_size = min(hero_stack, max(to_call * 2, pot * 0.5))
            if raise_size <= to_call:
                raise_size = min(hero_stack, to_call)

            if roll < 0.4:
                return "raise", raise_size
            elif roll < 0.8:
                return "call", None
            else:
                return "fold", None
        else:
            bet_fraction = random.uniform(0.4, 0.7)
            bet_amount = min(hero_stack, pot * bet_fraction)
            # If pot is zero (rare), bet small blind equivalent: use 1 as baseline
            if pot <= 0:
                bet_amount = min(hero_stack, max(1.0, hero_stack * 0.05))
            return "raise", bet_amount  # engine expects "raise" for bets per instructions


def main():
    if len(sys.argv) != 5:
        print("Usage: python maniac_bot.py <hero_id> <api_key> <table_name> <host:port>")
        print("Example: python maniac_bot.py p1 dev table-1 localhost:8080")
        sys.exit(1)

    hero_id = sys.argv[1]
    api_key = sys.argv[2]
    table = sys.argv[3]
    host = sys.argv[4]

    bot = ManiacBot(hero_id=hero_id, api_key=api_key, table=table, host=host)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
