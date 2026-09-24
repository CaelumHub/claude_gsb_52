"""In-memory transaction pool (mempool).

The pool holds signed, validated-but-unconfirmed transactions until a block is
mined.  It enforces:

* signature validity,
* sender authenticity (public key must match the sender address),
* nonce continuity (a sender may have a queue of consecutive nonces, preventing
  gaps and making double-spends structurally impossible),
* sufficient balance against the current world state plus queued transactions,
* fee >= 0 and a bounded pool size.

When a block is mined, included transactions are dropped; on a reorg or an
administrative rollback, the transactions from abandoned blocks are re-admitted
so they are not lost.  Each sender may have multiple queued transactions; only
the transactions that are executable with the current account nonce/balance are
packed into a block.
"""

from .config import TXPOOL_SORT_KEY
from .transaction import Transaction


class TxPool:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self._pool = {}              # txid -> Transaction
        self._order = []             # txids in arrival order
        self._by_sender = {}         # sender -> {nonce: txid}

    # ------------------------------------------------------------------ #
    # Access
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self._pool)

    def size(self):
        return len(self._pool)

    def get(self, txid):
        return self._pool.get(txid)

    def contains(self, txid):
        return txid in self._pool

    def all(self):
        return [self._pool[t] for t in self._order]

    def ready_all(self, world_state):
        """Return executable transactions in pool arrival order.

        A later transaction from one sender cannot be packed before the prior
        nonce has executed, and balance checks must account for the preceding
        queued transactions.  Walking the arrival order once therefore yields a
        valid block candidate list while preserving fairness between senders.
        """
        nonces = {}
        balances = {}
        ready = []
        for tx in self.all():
            sender = tx.sender
            current_nonce = nonces.get(
                sender, world_state.nonce(sender))
            if tx.nonce != current_nonce:
                continue

            balance = balances.get(sender, world_state.balance(sender))
            cost = tx.fee + (tx.amount if tx.tx_type in ("transfer", "call")
                             else 0.0)
            if balance < cost:
                continue

            ready.append(tx)
            nonces[sender] = current_nonce + 1
            balances[sender] = balance - cost
        return ready

    def ordered_all(self):
        """Transactions listed for the UI in display order."""
        return sorted(self.all(),
                      key=lambda tx: getattr(tx, TXPOOL_SORT_KEY, "") or "")

    def txids(self):
        return list(self._order)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _queued_nonces(self, sender):
        return set(self._by_sender.get(sender, {}))

    def _next_nonce(self, sender, world_state):
        """First nonce not already occupied by this sender's queued txs."""
        nonce = world_state.nonce(sender)
        queued = self._queued_nonces(sender)
        while nonce in queued:
            nonce += 1
        return nonce

    def _pending_cost(self, sender, world_state, exclude_txid=None):
        """Total balance reserved by the sender's contiguous queued prefix."""
        nonce = world_state.nonce(sender)
        queued = self._by_sender.get(sender, {})
        cost = 0.0
        while nonce in queued:
            txid = queued[nonce]
            if txid != exclude_txid:
                tx = self._pool[txid]
                cost += tx.fee
                if tx.tx_type in ("transfer", "call"):
                    cost += tx.amount
            nonce += 1
        return cost

    def validate(self, tx, world_state):
        """Return ``(ok, reason)`` for admitting ``tx`` into the pool."""
        if not isinstance(tx, Transaction):
            return False, "not a transaction"
        if tx.tx_type == "coinbase":
            return False, "coinbase transactions cannot be submitted to the pool"
        if tx.txid in self._pool:
            return False, "transaction already in pool"
        if not tx.validate_signature():
            return False, "invalid signature"
        if tx.derived_sender() != tx.sender:
            return False, "sender does not match public key"
        if tx.fee < 0 or tx.amount < 0:
            return False, "negative fee or amount"

        queued = self._queued_nonces(tx.sender)
        if tx.nonce in queued:
            return False, f"nonce {tx.nonce} already has a pending transaction"
        expected_nonce = self._next_nonce(tx.sender, world_state)
        if tx.nonce != expected_nonce:
            return False, (f"nonce {tx.nonce} != expected {expected_nonce} "
                           f"(next account nonce)")

        required = self._pending_cost(tx.sender, world_state) + tx.fee
        if tx.tx_type == "transfer":
            if not tx.to:
                return False, "transfer requires a recipient"
            required += tx.amount
            available = world_state.balance(tx.sender)
            if available < required:
                return False, "insufficient balance"
        elif tx.tx_type == "deploy":
            if world_state.balance(tx.sender) < required:
                return False, "insufficient balance for deploy fee"
        elif tx.tx_type == "call":
            required += tx.amount
            if world_state.balance(tx.sender) < required:
                return False, "insufficient balance for call"
        else:
            return False, f"unknown transaction type '{tx.tx_type}'"
        return True, "ok"

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    def add(self, tx, force=False):
        """Add a transaction to the pool.

        ``force`` is used when restoring transactions from rolled-back blocks:
        those transactions are already consensus history and must not be lost to
        the pool's normal capacity limit.
        """
        if tx.txid in self._pool:
            return False
        if tx.is_coinbase():
            return False
        if not force and tx.sender not in (None, ""):
            queued = self._by_sender.get(tx.sender, {})
            if tx.nonce in queued:
                return False

        if not force and self.size() >= self.max_size:
            if not self._evict_for(tx):
                return False

        self._pool[tx.txid] = tx
        self._order.append(tx.txid)
        if tx.sender:
            self._by_sender.setdefault(tx.sender, {})[tx.nonce] = tx.txid
        return True

    def _evict_for(self, tx):
        """Make room for ``tx`` without splitting a sender's nonce queue."""
        # Prefer to evict the oldest transaction belonging to another sender.
        for txid in list(self._order):
            candidate = self._pool[txid]
            if candidate.sender != tx.sender:
                self.remove(txid)
                return True

        # If every queued transaction belongs to this sender, evicting one would
        # create a nonce gap.  Restored transactions bypass the cap via force;
        # normal submissions must wait for the queued prefix to be mined.
        return False

    def remove(self, txid):
        if txid in self._pool:
            tx = self._pool.pop(txid, None)
            if tx is not None and tx.sender:
                queued = self._by_sender.get(tx.sender)
                if queued is not None:
                    queued.pop(tx.nonce, None)
                    if not queued:
                        self._by_sender.pop(tx.sender, None)
            if txid in self._order:
                self._order.remove(txid)
            return True
        return False

    def remove_many(self, txids):
        for txid in txids:
            self.remove(txid)

    def clear(self):
        self._pool.clear()
        self._order.clear()
        self._by_sender.clear()

    def re_admit(self, transactions):
        """Re-add transactions from abandoned/rolled-back blocks.

        Blocks are walked from lowest height to highest height, but callers may
        also pass a flat list.  Sort each sender's transactions by nonce so a
        multi-block sequence such as nonces 0, 1, 2 is restored in executable
        order.  Restored transactions bypass normal admission validation and the
        size cap because they came from valid blocks and must not be silently
        discarded.
        """
        txs = [tx for tx in transactions if not tx.is_coinbase()]
        txs.sort(key=lambda tx: (tx.sender, tx.nonce))
        added = []
        for tx in txs:
            queued = self._by_sender.get(tx.sender, {})
            if tx.txid in self._pool or tx.nonce in queued:
                continue
            if self.add(tx, force=True):
                added.append(tx)
        return added

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_list(self):
        return [tx.to_dict() for tx in self.all()]

    def load(self, data, world_state):
        self.clear()
        txs = [Transaction.from_dict(d) for d in (data or [])]
        txs.sort(key=lambda tx: (tx.sender, tx.nonce))
        for tx in txs:
            if self.validate(tx, world_state)[0]:
                self.add(tx, force=True)
