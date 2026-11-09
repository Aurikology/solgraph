"""Read-only Solana JSON-RPC, with retries and errors a beginner can act on.

This layer only ever calls read methods. There is no `sendTransaction` here and no private key.
Transfers are derived from each transaction's balance changes rather than from instruction
parsing, so they stay correct for swaps and program-routed transfers whose instructions this
code has never seen.
"""

from __future__ import annotations

import time
from collections import defaultdict

import httpx

from .cache import TTL, Cache, cache_key
from .models import (
    LAMPORTS_PER_SOL,
    SOL_MINT,
    TokenBalance,
    Transfer,
    TxRecord,
    WalletBalances,
)

TOKEN_PROGRAMS = (
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # SPL Token
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",  # Token-2022
)
SYSTEM_PROGRAM = "11111111111111111111111111111111"
DUST = 1e-9


class RpcError(RuntimeError):
    """Raised with a message meant to be shown to the user as-is."""


class SolanaRpc:
    def __init__(self, url: str, retries: int = 4, backoff: float = 0.8,
                 client: httpx.Client | None = None, cache: Cache | None = None):
        self.url = url
        self.cache = cache
        self.hits = self.misses = 0
        self.retries = retries
        self.backoff = backoff
        self.http = client or httpx.Client(timeout=20)
        self._id = 0

    def call(self, method: str, params: list | None = None):
        params = params or []
        key = cache_key(self.url, method, params) if self.cache and method in TTL else None
        if key:
            hit = self.cache.get(key)
            if hit is not None:
                self.hits += 1
                return hit
            self.misses += 1
        result = self._fetch(method, params)
        if key and result is not None:  # a transaction not found yet may appear later
            self.cache.set(key, result, TTL[method])
        return result

    def _fetch(self, method: str, params: list):
        self._id += 1
        body = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        last = ""
        for attempt in range(self.retries + 1):
            try:
                r = self.http.post(self.url, json=body)
            except httpx.TransportError as e:
                last = f"could not reach {self.url} ({type(e).__name__})"
            else:
                if r.status_code == 429 or r.status_code >= 500:
                    last = f"RPC returned HTTP {r.status_code}"
                else:
                    data = r.json()
                    if "error" in data:
                        raise RpcError(f"{method} failed: {data['error'].get('message', data['error'])}")
                    return data["result"]
            if attempt < self.retries:
                time.sleep(self.backoff * 2**attempt)
        raise RpcError(
            f"{method} failed after {self.retries + 1} tries: {last}. "
            "Public RPC endpoints rate-limit hard; wait a minute or set SOLGRAPH_RPC_URL "
            "to a private endpoint."
        )

    # --- reads -----------------------------------------------------------------------------

    def balances(self, address: str) -> WalletBalances:
        lamports = self.call("getBalance", [address])["value"]
        tokens: list[TokenBalance] = []
        for program in TOKEN_PROGRAMS:
            res = self.call("getTokenAccountsByOwner",
                            [address, {"programId": program}, {"encoding": "jsonParsed"}])
            for acct in res["value"]:
                info = acct["account"]["data"]["parsed"]["info"]
                amt = info["tokenAmount"]
                if float(amt.get("uiAmount") or 0) > 0:
                    tokens.append(TokenBalance(mint=info["mint"], amount=float(amt["uiAmount"]),
                                               decimals=int(amt["decimals"])))
        tokens.sort(key=lambda t: -t.amount)
        return WalletBalances(address=address, sol=lamports / LAMPORTS_PER_SOL, tokens=tokens)

    def signatures(self, address: str, limit: int = 20) -> list[dict]:
        return self.call("getSignaturesForAddress", [address, {"limit": min(limit, 1000)}])

    def transaction(self, signature: str) -> TxRecord | None:
        tx = self.call("getTransaction", [signature, {"encoding": "jsonParsed",
                                                      "maxSupportedTransactionVersion": 0}])
        return parse_transaction(signature, tx) if tx else None

    def account_kinds(self, addresses) -> dict[str, str]:
        """Who is behind each address: a person's wallet, a program, or an account a program
        owns (a pool vault, a PDA, a token account). One `getMultipleAccounts` call per 100
        addresses, asking for zero bytes of account data."""
        addrs = sorted(set(addresses))  # sorted: the same set always makes the same cache key
        out: dict[str, str] = {}
        for i in range(0, len(addrs), 100):
            chunk = addrs[i:i + 100]
            res = self.call("getMultipleAccounts",
                            [chunk, {"encoding": "base64", "dataSlice": {"offset": 0, "length": 0}}])
            for addr, acct in zip(chunk, res["value"]):
                out[addr] = classify_account(acct)
        return out


def classify_account(acct: dict | None) -> str:
    """`user` = owned by the System Program, so a keypair controls it. `program` = executable
    code. `program_account` = state some program controls. `unknown` = no account on chain
    (closed, or never funded)."""
    if acct is None:
        return "unknown"
    if acct.get("executable"):
        return "program"
    return "user" if acct.get("owner") == SYSTEM_PROGRAM else "program_account"


def parse_transaction(signature: str, tx: dict) -> TxRecord:
    """Turn a jsonParsed transaction into owner-to-owner transfers.

    Per asset, owners whose balance fell are paired with owners whose balance rose, largest
    first. The fee payer's SOL change has the fee added back first, so a fee is never reported
    as a transfer.
    """
    meta = tx["meta"]
    msg = tx["transaction"]["message"]
    keys = [k["pubkey"] if isinstance(k, dict) else k for k in msg["accountKeys"]]
    signers = [k["pubkey"] for k in msg["accountKeys"] if isinstance(k, dict) and k.get("signer")]
    fee = meta["fee"]

    deltas: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for i, (pre, post) in enumerate(zip(meta["preBalances"], meta["postBalances"])):
        change = post - pre + (fee if i == 0 else 0)
        if change:
            deltas[SOL_MINT][keys[i]] += change / LAMPORTS_PER_SOL

    def ui(b: dict) -> float:
        return float(b["uiTokenAmount"].get("uiAmount") or 0)

    for b in meta.get("preTokenBalances") or []:
        if b.get("owner"):
            deltas[b["mint"]][b["owner"]] -= ui(b)
    for b in meta.get("postTokenBalances") or []:
        if b.get("owner"):
            deltas[b["mint"]][b["owner"]] += ui(b)

    # A token account's rent shows up as a SOL drop with no SOL receiver among owners; the
    # pairing below just leaves it unmatched rather than inventing a recipient.
    transfers: list[Transfer] = []
    for mint, by_owner in deltas.items():
        out = sorted(((o, -d) for o, d in by_owner.items() if d < -DUST), key=lambda x: -x[1])
        inc = sorted(((o, d) for o, d in by_owner.items() if d > DUST), key=lambda x: -x[1])
        i = j = 0
        while i < len(out) and j < len(inc):
            src, have = out[i]
            dst, need = inc[j]
            moved = min(have, need)
            if src != dst and moved > DUST:
                transfers.append(Transfer(signature=signature, source=src, destination=dst,
                                          mint=mint, amount=round(moved, 9),
                                          block_time=tx.get("blockTime")))
            out[i] = (src, have - moved)
            inc[j] = (dst, need - moved)
            if out[i][1] <= DUST:
                i += 1
            if inc[j][1] <= DUST:
                j += 1

    return TxRecord(signature=signature, slot=tx["slot"], block_time=tx.get("blockTime"),
                    fee_sol=fee / LAMPORTS_PER_SOL, success=meta.get("err") is None,
                    signers=signers, transfers=transfers)
