"""
poker_bot.py

Texas Hold'em poker bot skeleton for AI class.
- Hand representation & evaluation
- Monte Carlo win probability estimation
- Simple opponent modeling
- EV-based decision policy
- WebSocket skeleton for engine integration

You will need to:
    pip install websockets
and adapt the WebSocket message formats (marked with TODOs).
"""

import asyncio
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

# ===============================
# Card representation & utilities
# ===============================

RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs, diamonds, hearts, spades
RANK_TO_INT = {r: i for i, r in enumerate(RANKS, start=2)}
INT_TO_RANK = {v: k for k, v in RANK_TO_INT.items()}


@dataclass(frozen=True)
class Card:
    rank: int  # 2-14
    suit: str  # 'c', 'd', 'h', 's'


def parse_card(cs: str) -> Card:
    """
    Parse a 2-char card string like 'Ah', 'Tc', '7d'.
    Assumes valid input.
    """
    rank_char = cs[0].upper()
    suit_char = cs[1].lower()
    return Card(RANK_TO_INT[rank_char], suit_char)


def card_str(card: Card) -> str:
    return f"{INT_TO_RANK[card.rank]}{card.suit}"


def full_deck() -> list[Card]:
    return [Card(RANK_TO_INT[r], s) for r in RANKS for s in SUITS]


# ==========================
# 5-card hand strength score
# ==========================

def evaluate_5cards(cards: list[Card]) -> tuple:
    """
    Return a rank tuple for 5-card poker hand.
    Higher tuple compares as stronger.

    Returns: (category, tiebreakers...)
    category:
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
    # Sort by (count, rank) descending
    count_rank = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    is_flush = len({c.suit for c in cards}) == 1

    # Straight detection (handle A-5 wheel)
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    high_straight = None
    if len(unique_ranks) >= 5:
        # Normal straight
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            high_straight = unique_ranks[0]
        # Wheel: A-5 (A,5,4,3,2 => 14,5,4,3,2)
        elif unique_ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            high_straight = 5

    # Straight flush?
    if is_flush and is_straight:
        return (8, high_straight)

    # Four of a kind / full house / trips / pairs logic using counts
    # count_rank is like [(rank1, count1), (rank2, count2), ...] sorted by count then rank
    (r1, c1), (r2, c2), *rest = count_rank

    # Four of a kind
    if c1 == 4:
        kicker = max(r for r in ranks if r != r1)
        return (7, r1, kicker)

    # Full house (3 + 2)
    if c1 == 3 and c2 >= 2:
        return (6, r1, r2)

    # Flush
    if is_flush:
        return (5, *sorted(ranks, reverse=True))

    # Straight
    if is_straight:
        return (4, high_straight)

    # Three of a kind
    if c1 == 3:
        kickers = sorted([r for r in ranks if r != r1], reverse=True)
        return (3, r1, *kickers)

    # Two pair
    if c1 == 2 and c2 == 2:
        pair1, pair2 = sorted([r1, r2], reverse=True)
        kicker = max(r for r in ranks if r != pair1 and r != pair2)
        return (2, pair1, pair2, kicker)

    # One pair
    if c1 == 2:
        pair = r1
        kickers = sorted([r for r in ranks if r != pair], reverse=True)
        return (1, pair, *kickers)

    # High card
    return (0, *sorted(ranks, reverse=True))


def evaluate_7cards(cards: list[Card]) -> tuple:
    """
    Best 5-card hand out of 7.
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
    hole_cards: list[Card],
    board_cards: list[Card],
    num_opponents: int,
    num_samples: int = 1000
) -> float:
    """
    Monte Carlo estimate of win probability for hero.

    Returns:
        equity in [0, 1] (ties count as 0.5 win)
    """
    used = set(hole_cards + board_cards)
    deck = [c for c in full_deck() if c not in used]

    if num_opponents <= 0:
        # Heads-up vs nobody (should not happen, but just in case)
        return 1.0

    wins = 0
    ties = 0

    for _ in range(num_samples):
        random.shuffle(deck)
        # Need remaining board cards + opponent hole cards
        needed_board = 5 - len(board_cards)
        cards_needed = needed_board + 2 * num_opponents
        sample = deck[:cards_needed]

        sample_board = board_cards + sample[:needed_board]
        opp_holes = [
            sample[needed_board + 2 * i: needed_board + 2 * (i + 1)]
            for i in range(num_opponents)
        ]

        hero_score = evaluate_7cards(hole_cards + sample_board)
        opp_scores = [evaluate_7cards(opp + sample_board) for opp in opp_holes]

        better = sum(1 for s in opp_scores if s > hero_score)
        equal = sum(1 for s in opp_scores if s == hero_score)

        if better == 0 and equal == 0:
            wins += 1
        elif better == 0 and equal > 0:
            ties += 1
        # otherwise, hero loses

    total = num_samples
    return (wins + 0.5 * ties) / total


# ===================
# Opponent Modeling
# ===================

@dataclass
class OpponentStats:
    hands_played: int = 0      # total hands seen
    vpip: int = 0              # voluntarily put money in pot (call/raise preflop)
    preflop_raises: int = 0
    postflop_raises: int = 0
    calls: int = 0
    folds: int = 0

    def vpip_rate(self) -> float:
        return self.vpip / self.hands_played if self.hands_played > 0 else 0.0

    def pfr_rate(self) -> float:
        return self.preflop_raises / self.hands_played if self.hands_played > 0 else 0.0

    def aggression_factor(self) -> float:
        # simple AF = raises / calls
        return self.postflop_raises / self.calls if self.calls > 0 else 0.0


class OpponentModel:
    def __init__(self):
        self.stats: dict[str, OpponentStats] = defaultdict(OpponentStats)

    def record_hand_start(self, players: list[str]):
        for pid in players:
            self.stats[pid].hands_played += 1

    def record_action(self, player_id: str, street: str, action: str):
        """
        action: 'fold', 'call', 'raise', 'check', 'bet'
        street: 'preflop', 'flop', 'turn', 'river'
        """
        s = self.stats[player_id]
        if street == "preflop":
            if action in ("call", "raise"):
                s.vpip += 1
            if action == "raise":
                s.preflop_raises += 1
        else:
            if action == "raise" or action == "bet":
                s.postflop_raises += 1

        if action == "fold":
            s.folds += 1
        elif action == "call":
            s.calls += 1

    def classify_player(self, player_id: str) -> str:
        """
        Rough labels: 'tight', 'loose', 'aggro', 'passive', 'unknown'
        """
        s = self.stats[player_id]
        if s.hands_played < 10:
            return "unknown"

        vpip = s.vpip_rate()
        pfr = s.pfr_rate()
        af = s.aggression_factor()

        if vpip < 0.15:
            return "tight"
        elif vpip > 0.35:
            return "loose"

        # Medium VPIP: differentiate by aggression
        if af > 2.0 or pfr > 0.25:
            return "aggro"
        else:
            return "passive"


# =====================
# Game state & strategy
# =====================

@dataclass
class GameState:
    # Minimal information needed by strategy
    hero_id: str
    hole_cards: list[Card]
    board_cards: list[Card]  # can be 0,3,4,5 cards
    pot_size: float
    to_call: float          # amount hero needs to call
    min_raise: float        # minimum legal raise size (total bet amount, not increment) – adapt to engine
    hero_stack: float       # chips behind
    num_active_opponents: int
    street: str             # 'preflop','flop','turn','river'
    opponent_ids: list[str] # for modeling


class Strategy:
    def __init__(self, opp_model: OpponentModel):
        self.opp_model = opp_model

    # --- helper methods ---

    def compute_pot_odds(self, state: GameState) -> float:
        """
        Pot odds = cost_to_call / (pot + cost_to_call)
        """
        if state.to_call <= 0:
            return 0.0
        return state.to_call / (state.pot_size + state.to_call)

    def choose_action(self, state: GameState) -> dict:
        """
        Main decision function.

        Returns a dict like:
            {"action": "fold" | "call" | "raise", "amount": optional_float}
        You will map this to your engine's protocol.
        """
        # 1) Estimate equity via Monte Carlo
        equity = estimate_equity(
            hole_cards=state.hole_cards,
            board_cards=state.board_cards,
            num_opponents=state.num_active_opponents,
            num_samples=800  # adjust for speed
        )

        pot_odds = self.compute_pot_odds(state)

        # Simple opponent-based aggression adjustment:
        # if table seems tight, we can bluff more (effectively treat equity as a bit higher)
        tight_count = sum(
            1 for pid in state.opponent_ids
            if self.opp_model.classify_player(pid) == "tight"
        )
        loose_count = sum(
            1 for pid in state.opponent_ids
            if self.opp_model.classify_player(pid) == "loose"
        )

        # Adjusted equity for decision purposes
        adj_equity = equity
        adj_equity += 0.05 * (tight_count > loose_count)  # small bump vs tight tables
        adj_equity = max(0.0, min(1.0, adj_equity))

        # 2) Preflop special handling (simple hand chart)
        if state.street == "preflop":
            return self.preflop_decision(state, adj_equity)

        # 3) Postflop: compare equity vs pot odds with some margins
        # Basic rules:
        #   - If adj_equity < pot_odds - margin -> fold
        #   - If adj_equity slightly > pot_odds -> call
        #   - If adj_equity >> pot_odds -> raise for value

        margin = 0.05  # safety margin
        if state.to_call > 0:
            if adj_equity + margin < pot_odds:
                return {"action": "fold"}

            # Calling region
            # If equity not super strong, just call
            if adj_equity < pot_odds + 0.15:
                return {"action": "call"}

            # Strong hand -> raise
            # Simple rule: raise to around pot size (adapt to engine rules)
            raise_size = min(state.hero_stack, state.pot_size * 0.75 + state.to_call)
            if raise_size <= state.to_call:  # can't raise meaningfully
                return {"action": "call"}
            return {"action": "raise", "amount": raise_size}
        else:
            # No bet to us (we can check or bet)
            # If equity is low, check.
            # If decent, semi-bluff occasionally.
            if adj_equity < 0.35:
                return {"action": "check"}
            elif adj_equity < 0.55:
                # Semi-bluff region: bet sometimes
                if random.random() < 0.4:
                    bet_size = min(state.hero_stack, state.pot_size * 0.5)
                    return {"action": "bet", "amount": bet_size}
                else:
                    return {"action": "check"}
            else:
                # Strong equity: value bet
                bet_size = min(state.hero_stack, state.pot_size * 0.75)
                return {"action": "bet", "amount": bet_size}

    def preflop_decision(self, state: GameState, adj_equity: float) -> dict:
        """
        Simple preflop strategy; can be improved with real hand charts.
        For now we use:
            - premium hands: raise
            - decent hands: call / raise depending on action
            - junk: mostly fold, sometimes bluff
        """
        c1, c2 = state.hole_cards
        ranks_sorted = sorted([c1.rank, c2.rank], reverse=True)
        same_suit = c1.suit == c2.suit
        high_card = ranks_sorted[0]
        low_card = ranks_sorted[1]

        # Quick classification
        is_pair = c1.rank == c2.rank
        is_premium_pair = is_pair and c1.rank >= RANK_TO_INT["T"]  # TT+
        is_mid_pair = is_pair and RANK_TO_INT["6"] <= c1.rank <= RANK_TO_INT["9"]

        # Broadways or strong suited connectors
        is_broadway = high_card >= RANK_TO_INT["T"]
        gap = high_card - low_card

        # Very rough preflop rules
        if is_premium_pair or (is_broadway and same_suit and gap <= 3):
            # Premium: raise or 3-bet
            if state.to_call == 0:
                # Open-raise ~3x BB; here we just raise pot-ish
                raise_size = min(state.hero_stack, max(state.min_raise, state.pot_size * 0.5 + 1))
                return {"action": "raise", "amount": raise_size}
            else:
                # Facing a raise: 3-bet
                raise_size = min(state.hero_stack, state.to_call * 3)
                return {"action": "raise", "amount": raise_size}

        if is_mid_pair or (is_broadway and gap <= 4):
            # Playable hand
            if state.to_call == 0:
                # open raise often
                if random.random() < 0.8:
                    raise_size = min(state.hero_stack, max(state.min_raise, state.pot_size * 0.5 + 1))
                    return {"action": "raise", "amount": raise_size}
                else:
                    return {"action": "check"}
            else:
                # facing a raise: call if equity roughly ok
                pot_odds = self.compute_pot_odds(state)
                if adj_equity >= pot_odds - 0.05:
                    return {"action": "call"}
                else:
                    return {"action": "fold"}

        # Junk hands: fold to raises, occasionally bluff-raise if unopened
        if state.to_call > 0:
            return {"action": "fold"}
        else:
            # Sometimes open with a bluff if folded to us
            if random.random() < 0.15:
                raise_size = min(state.hero_stack, max(state.min_raise, state.pot_size * 0.4 + 1))
                return {"action": "raise", "amount": raise_size}
            else:
                return {"action": "check"}


# =======================
# WebSocket Client Skeleton
# =======================

import websockets  # pip install websockets


class PokerBot:
    def __init__(self, hero_id: str, uri: str):
        self.hero_id = hero_id
        self.uri = uri
        self.opp_model = OpponentModel()
        self.strategy = Strategy(self.opp_model)

        # Keep some state from engine
        self.current_players: list[str] = []

    async def run(self):
        async with websockets.connect(self.uri) as ws:
            while True:
                msg = await ws.recv()
                data = json.loads(msg)

                # TODO: adapt these branches to your engine protocol
                event_type = data.get("type")

                if event_type == "hand_start":
                    self.handle_hand_start(data)
                elif event_type == "player_action":
                    self.handle_player_action(data)
                elif event_type == "your_turn":
                    action_msg = self.handle_your_turn(data)
                    await ws.send(json.dumps(action_msg))
                elif event_type == "hand_end":
                    self.handle_hand_end(data)
                else:
                    # unknown / heartbeat etc.
                    pass

    def handle_hand_start(self, data: dict):
        """
        Expect something like:
            {
              "type": "hand_start",
              "players": ["p1","p2","me",...],
              ...
            }
        """
        players = data["players"]
        self.current_players = players
        self.opp_model.record_hand_start(players)

    def handle_player_action(self, data: dict):
        """
        Expect something like:
            {
              "type": "player_action",
              "player_id": "p2",
              "street": "flop",
              "action": "raise",   # fold/call/raise/check/bet
              ...
            }
        """
        pid = data["player_id"]
        street = data["street"]
        action = data["action"]
        if pid != self.hero_id:
            self.opp_model.record_action(pid, street, action)

    def handle_your_turn(self, data: dict) -> dict:
        """
        Expect something like:
            {
              "type": "your_turn",
              "hero_id": "...",
              "hole_cards": ["Ah","Kd"],
              "board": ["Ts","Jh","2c"],
              "pot": 120.0,
              "to_call": 20.0,
              "min_raise": 40.0,
              "stack": 350.0,
              "street": "flop",
              "active_opponents": ["p1","p2"]
            }
        Returns dict with action to send back, e.g.
            {"type": "action", "action": "call"}
        or   {"type": "action", "action": "raise", "amount": 80.0}
        """
        hole = [parse_card(c) for c in data["hole_cards"]]
        board = [parse_card(c) for c in data["board"]]

        state = GameState(
            hero_id=self.hero_id,
            hole_cards=hole,
            board_cards=board,
            pot_size=float(data["pot"]),
            to_call=float(data["to_call"]),
            min_raise=float(data["min_raise"]),
            hero_stack=float(data["stack"]),
            num_active_opponents=len(data["active_opponents"]),
            street=data["street"],
            opponent_ids=data["active_opponents"],
        )

        decision = self.strategy.choose_action(state)

        # For debugging / explainability
        print("=== MY TURN ===")
        print("Hole:", [card_str(c) for c in hole],
              "Board:", [card_str(c) for c in board])
        print("Decision:", decision)

        # Map to engine message format
        action_msg = {"type": "action", "action": decision["action"]}
        if "amount" in decision:
            action_msg["amount"] = decision["amount"]
        return action_msg

    def handle_hand_end(self, data: dict):
        """
        Hook if you want to do any hand-end logging, etc.
        """
        pass


# =================================
# Entry point (for local testing)
# =================================

if __name__ == "__main__":
    # Example:
    #   python poker_bot.py ws://localhost:8000/game
    import sys

    if len(sys.argv) != 3:
        print("Usage: python poker_bot.py <hero_id> <ws_uri>")
        print("Example: python poker_bot.py bot1 ws://localhost:8000/game")
        sys.exit(1)

    hero_id = sys.argv[1]
    uri = sys.argv[2]

    bot = PokerBot(hero_id=hero_id, uri=uri)
    asyncio.run(bot.run())
