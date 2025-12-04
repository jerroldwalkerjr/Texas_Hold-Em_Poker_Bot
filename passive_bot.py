"""
passive_bot.py

Passive calling-station bot for the WebSocket Texas Hold'em engine.
Behavior:
- Preflop: call when facing a bet, check when possible, never raise.
- Postflop: call bets up to 25% of current stack, otherwise fold; check when free.

Usage:
    pip install websockets
    python passive_bot.py <hero_id> <api_key> <table_name> <host:port>
"""

import asyncio
import json
import sys
from typing import Dict, List

import websockets

# Card helpers (copied from poker_bot.py for compatibility with engine JSON)
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


class PassiveBot:
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
                    # Ignore unknown messages but keep the loop alive.
                    pass

    async def handle_state(self, ws, data: Dict):
        """
        Map engine JSON (same assumptions as poker_bot.py) and choose a passive action.
        Expected fields (adjust if your engine differs):
            players: list of player dicts with id/stack/bet/inHand/cards
            turn: id of player to act
            pot: total pot
            callAmount: amount required to call
            community: list of board cards
            phase: preflop/flop/turn/river
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

        # Decide action based on phase and bet size
        action = "check"
        if phase == "preflop":
            action = "call" if to_call > 0 else "check"
        else:
            if to_call <= 0:
                action = "check"
            else:
                threshold = 0.25 * hero_stack
                action = "call" if to_call <= threshold else "fold"

        # Debug output to mirror engine state and choice
        print("=== MY TURN ===")
        print("Phase:", phase.upper())
        print("Hole:", [card_str(c) for c in self.hole_cards])
        print("Board:", [card_str(c) for c in board_cards])
        print(f"Pot: {pot} To call: {to_call} Stack: {hero_stack}")
        print("Action:", action)

        msg = {"type": "action", "action": action}
        await ws.send(json.dumps(msg))


def main():
    if len(sys.argv) != 5:
        print("Usage: python passive_bot.py <hero_id> <api_key> <table_name> <host:port>")
        print("Example: python passive_bot.py p1 dev table-1 localhost:8080")
        sys.exit(1)

    hero_id = sys.argv[1]
    api_key = sys.argv[2]
    table = sys.argv[3]
    host = sys.argv[4]

    bot = PassiveBot(hero_id=hero_id, api_key=api_key, table=table, host=host)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
