"""In-memory transaction pool (mempool).

The pool holds signed, validated-but-unconfirmed transactions until a block is
mined.  It enforces:

* signature validity,
* sender authenticity (public key must match the sender address),
* nonce monotonicity (only the *next* nonce per sender is admitted, preventing
  nonce gaps and making double-spends structurally impossible),
* sufficient balance against the current world state,
* fee >= 0 and a bounded pool size.

Only the transaction carrying a sender's next expected nonce lives in the
main pool (one mineable pending tx per sender).  Transactions with a higher
nonce — which appear when a rollback or re-org re-admits several transactions
of the same sender from abandoned blocks — are kept in a per-sender *queue*
keyed by nonce instead of being dropped.  Whenever the account nonce advances
(a block is mined/received), :meth:`promote` moves queued transactions whose
nonce has become current into the main pool, so re-admitted transactions are
never silently lost.

When a block is mined, included transactions are dropped; on a reorg, the
transactions from abandoned blocks are re-admitted so they are not lost.
"""

from .config import TXPOOL_SORT_KEY
from .transaction import Transaction


class TxPool:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self._pool = {}          # txid -> Transaction
        self._order = []         # txids in arrival order
        self._by_sender = {}     # sender -> txid (one pending tx per sender)
        # sender -> {nonce: Transaction}; higher-nonce txs awaiting their turn
        self._queued = {}

    # ------------------------------------------------------------------ #
    # Access
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self._pool)

    def size(self):
        return len(self._pool)

    def queued_size(self):
        return sum(len(q) for q in self._queued.values())

    def get(self, txid):
        if txid in self._pool:
            return self._pool[txid]
        for q in self._queued.values():
            for tx in q.values():
                if tx.txid == txid:
                    return tx
        return None

    def contains(self, txid):
        if txid in self._pool:
            return True
        return any(tx.txid == txid
                   for q in self._queued.values() for tx in q.values())

    def all(self):
        return [self._pool[t] for t in self._order]

    def queued_all(self):
        """Future-nonce transactions parked in sender queues (nonce order)."""
        result = []
        for q in self._queued.values():
            result.extend(q[nonce] for nonce in sorted(q))
        return result

    def ordered_all(self):
        """Transactions listed for the UI in display order."""
        return sorted(self.all(),
                      key=lambda tx: getattr(tx, TXPOOL_SORT_KEY, "") or "")

    def txids(self):
        return list(self._order)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self, tx, world_state):
        """Return ``(ok, reason)`` for admitting ``tx`` into the pool."""
        if not isinstance(tx, Transaction):
            return False, "not a transaction"
        if tx.tx_type == "coinbase":
            return False, "coinbase transactions cannot be submitted to the pool"
        if self.contains(tx.txid):
            return False, "transaction already in pool"
        if not tx.validate_signature():
            return False, "invalid signature"
        if tx.derived_sender() != tx.sender:
            return False, "sender does not match public key"
        if tx.sender in self._by_sender:
            return False, "sender already has a pending transaction"
        if tx.fee < 0 or tx.amount < 0:
            return False, "negative fee or amount"
        expected_nonce = world_state.nonce(tx.sender)
        if tx.nonce != expected_nonce:
            return False, (f"nonce {tx.nonce} != expected {expected_nonce} "
                           f"(account nonce)")
        if tx.tx_type == "transfer":
            if not tx.to:
                return False, "transfer requires a recipient"
            if world_state.balance(tx.sender) < tx.amount + tx.fee:
                return False, "insufficient balance"
        elif tx.tx_type == "deploy":
            if world_state.balance(tx.sender) < tx.fee:
                return False, "insufficient balance for deploy fee"
        elif tx.tx_type == "call":
            if world_state.balance(tx.sender) < tx.amount + tx.fee:
                return False, "insufficient balance for call"
        else:
            return False, f"unknown transaction type '{tx.tx_type}'"
        return True, "ok"

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    def add(self, tx):
        if tx.txid in self._pool:
            return False
        if tx.sender in self._by_sender and tx.sender not in (None, ""):
            return False
        if self.size() >= self.max_size:
            # Evict the oldest transaction to stay within bounds.
            oldest = self._order.pop(0)
            evicted = self._pool.pop(oldest, None)
            if evicted is not None:
                self._by_sender.pop(evicted.sender, None)
        self._pool[tx.txid] = tx
        self._order.append(tx.txid)
        if tx.sender:
            self._by_sender[tx.sender] = tx.txid
        return True

    def remove(self, txid):
        if txid in self._pool:
            tx = self._pool.pop(txid, None)
            if tx is not None and tx.sender:
                self._by_sender.pop(tx.sender, None)
            if txid in self._order:
                self._order.remove(txid)
            return True
        # May be parked in the future-nonce queue.
        for sender, q in list(self._queued.items()):
            for nonce, tx in q.items():
                if tx.txid == txid:
                    q.pop(nonce, None)
                    if not q:
                        self._queued.pop(sender, None)
                    return True
        return False

    def remove_many(self, txids):
        for txid in txids:
            self.remove(txid)

    def clear(self):
        self._pool.clear()
        self._order.clear()
        self._by_sender.clear()
        self._queued.clear()

    def re_admit(self, transactions, world_state=None):
        """Re-add transactions (e.g. from abandoned/rolled-back blocks).

        Transactions whose nonce is current (and which satisfy the normal
        admission checks when ``world_state`` is given) go straight into the
        main pool.  Transactions carrying a *future* nonce — produced when one
        sender had transactions in several abandoned blocks — are parked in the
        per-sender queue rather than dropped, and promoted once earlier nonces
        are mined again.  Transactions whose nonce is already below the account
        nonce are stale (already executed) and discarded.
        """
        for tx in transactions:
            if tx.is_coinbase() or self.contains(tx.txid):
                continue
            if not self._statically_valid(tx):
                continue
            expected = (world_state.nonce(tx.sender) if world_state is not None
                        else tx.nonce)
            if tx.nonce < expected:
                # Already applied on the current chain; nothing to re-admit.
                continue
            if tx.nonce == expected and tx.sender not in self._by_sender:
                if world_state is None or self.validate(tx, world_state)[0]:
                    self.add(tx)
                    continue
            # Future nonce (or not currently admissible, e.g. insufficient
            # balance): park it for later promotion instead of dropping it.
            self._enqueue(tx)

    def _enqueue(self, tx):
        """Park a higher-nonce transaction in its sender's nonce queue."""
        if not tx.sender:
            return
        q = self._queued.setdefault(tx.sender, {})
        # First transaction for a given (sender, nonce) wins; same body means
        # the same txid, and a different body is a conflicting replacement.
        if tx.nonce in q:
            return
        if self.queued_size() >= self.max_size:
            return
        q[tx.nonce] = tx

    def _statically_valid(self, tx):
        """Checks that never change while a tx sits in the queue."""
        if tx.is_coinbase():
            return False
        if not tx.validate_signature():
            return False
        if tx.derived_sender() != tx.sender:
            return False
        if tx.fee < 0 or tx.amount < 0:
            return False
        return tx.tx_type in ("transfer", "deploy", "call")

    def promote(self, world_state):
        """Move queued transactions whose nonce is now current into the pool.

        Called after each committed block (the account nonces move forward).
        A sender's queue is promoted in nonce order, one transaction at a time,
        so the strict "next nonce only" invariant of the main pool is preserved.
        Stale entries (nonce already consumed) and permanently invalid txs are
        dropped; a tx that is only temporarily inadmissible (insufficient
        balance) stays parked and is retried after a later block.
        """
        for sender, q in list(self._queued.items()):
            while q:
                expected = world_state.nonce(sender)
                head_nonce = min(q)
                if head_nonce < expected:
                    q.pop(head_nonce, None)
                    continue
                if head_nonce > expected or sender in self._by_sender:
                    break
                tx = q.pop(head_nonce)
                if not self._statically_valid(tx):
                    continue  # permanently invalid: discard, unblock the queue
                if self.validate(tx, world_state)[0]:
                    self.add(tx)
                    continue
                # Transient failure (insufficient balance): park it again and
                # wait for a later promotion instead of dropping it; later
                # nonces cannot pass it, so stop processing this sender.
                q[head_nonce] = tx
                break
            if not q:
                self._queued.pop(sender, None)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_list(self):
        pending = [tx.to_dict() for tx in self.all()]
        queued = [tx.to_dict()
                  for q in self._queued.values()
                  for nonce in sorted(q)
                  for tx in [q[nonce]]]
        return {"pending": pending, "queued": queued}

    def load(self, data, world_state):
        self.clear()
        # Backwards compatibility: older txpool files were a bare list.
        if isinstance(data, dict):
            pending = data.get("pending") or []
            queued = data.get("queued") or []
        else:
            pending, queued = data or [], []
        for d in pending:
            tx = Transaction.from_dict(d)
            if self.validate(tx, world_state)[0]:
                self.add(tx)
            elif not tx.is_coinbase() and tx.nonce >= world_state.nonce(tx.sender):
                self._enqueue(tx)
        for d in queued:
            tx = Transaction.from_dict(d)
            if not tx.is_coinbase() and tx.nonce >= world_state.nonce(tx.sender):
                self._enqueue(tx)
        self.promote(world_state)
