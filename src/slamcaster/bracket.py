 from __future__ import annotations
 
 from dataclasses import dataclass
 from typing import Iterable
 
 
 @dataclass(frozen=True)
 class DrawEntry:
     slot: int
     player_name: str
     seed: int | None = None
     player_id: int | None = None
 
 
 def validate_power_of_two(n: int) -> None:
     if n <= 1 or (n & (n - 1)) != 0:
         raise ValueError(f"draw_size must be a power of two, got {n}")
 
 
 def first_round_pairs(entries: list[DrawEntry]) -> list[tuple[DrawEntry, DrawEntry]]:
     """Assumes entries are ordered by slot. Pair (1,2), (3,4), ..."""
     if len(entries) % 2 != 0:
         raise ValueError("draw size must be even")
     pairs = []
     for i in range(0, len(entries), 2):
         pairs.append((entries[i], entries[i + 1]))
     return pairs
