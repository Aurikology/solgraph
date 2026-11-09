"""Validated shapes for everything that crosses a boundary: AI tool input, RPC output, graph rows.

Every address the AI hands a tool goes through `Address` first, so a hallucinated or malicious
string never reaches the RPC or, more importantly, a Cypher query.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

BASE58 = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
SOL_MINT = "SOL"  # native SOL has no mint account; the graph uses this marker instead
LAMPORTS_PER_SOL = 1_000_000_000
# What an address is, from its on-chain owner (see rpc.classify_account). Traces stop at the
# two program kinds: walking back through a DEX vault would fan out into unrelated users.
KINDS = ("user", "program", "program_account", "unknown")
PROGRAM_KINDS = frozenset({"program", "program_account"})


def _check_address(v: str) -> str:
    v = v.strip()
    if not BASE58.fullmatch(v):
        raise ValueError(
            f"{v!r} is not a Solana address: expected 32-44 base58 characters "
            "(no 0, O, I or l)."
        )
    return v


def _check_mint(v: str) -> str:
    return v if v == SOL_MINT else _check_address(v)


Address = Annotated[str, AfterValidator(_check_address)]
Mint = Annotated[str, AfterValidator(_check_mint)]


class TokenBalance(BaseModel):
    mint: Mint
    amount: float = Field(description="UI amount, already divided by 10**decimals")
    decimals: int


class WalletBalances(BaseModel):
    address: Address
    sol: float
    tokens: list[TokenBalance]


class Transfer(BaseModel):
    """One movement of one asset between two owners, derived from balance changes."""

    signature: str
    source: Address
    destination: Address
    mint: Mint
    amount: float
    block_time: int | None = None


class TxRecord(BaseModel):
    signature: str
    slot: int
    block_time: int | None
    fee_sol: float
    success: bool
    signers: list[Address]
    transfers: list[Transfer]
