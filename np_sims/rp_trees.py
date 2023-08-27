from typing import Optional
import pickle
from time import perf_counter
from math import log2, ceil

import numpy as np
from np_sims.partition import rptree_max_chooserule


DEFAULT_DEPTH = 5


def _fit(vectors: np.ndarray, depth: int = DEFAULT_DEPTH):
    """Build a random projection tree from a set of vectors."""
    # TODO sample before calling this function
    # Sample N vectors, get the two with smallest dot product
    # between them
    N = 100000
    sample = vectors
    if len(vectors) > N:
        sample = vectors[np.random.choice(len(vectors), N, replace=False)]
    splitter = rptree_max_chooserule(sample)
    lhs_idx, rhs_idx = splitter.split(vectors)

    if depth > 1:
        left = _fit(vectors[lhs_idx], depth - 1) if len(lhs_idx) > 1 else None
        right = _fit(vectors[rhs_idx], depth - 1) if len(rhs_idx) > 1 else None

        return splitter, left, right
    else:
        return splitter, None, None


def _rp_hash(tree,
             vectors: np.ndarray[np.float64],
             hashes: np.ndarray[np.uint64],
             depth: int = DEFAULT_DEPTH):
    """Walk a vector down a tree to get a hash."""
    if tree is None or depth <= 0:
        return hashes

    root, left, right = tree
    lhs_idx, rhs_idx = root.split(vectors)

    lhs_vectors = vectors[lhs_idx]
    rhs_vectors = vectors[rhs_idx]

    lhs_hashes = hashes[lhs_idx]
    rhs_hashes = hashes[rhs_idx]

    hashes[lhs_idx] |= _rp_hash(left, lhs_vectors, lhs_hashes, depth - 1)
    # Set the 'depth' bit in rhs_idx
    hashes[rhs_idx] |= (1 << depth) | (_rp_hash(right, rhs_vectors, rhs_hashes, depth - 1))

    return hashes


def _hash_table(hashes):
    """Sort hashes, produce an array of the sorts with the actual values."""
    sorted_idxs = np.argsort(hashes, kind="stable")
    sorted_hashes = hashes[sorted_idxs]
    return sorted_idxs, sorted_hashes


class RandomProjectionTree:
    """Pure Python random projection tree implementation."""

    def __init__(self, root, sorted_hashes, sorted_idxs):
        self.sorted_hashes = sorted_hashes
        self.sorted_idxs = sorted_idxs
        self.root = root

    def save(self, path: str):
        """Save the tree to a file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def build(vectors: np.ndarray, seed: Optional[int] = None):
        """Build a random projection tree from a set of vectors."""
        start = perf_counter()
        if seed is not None:
            np.random.seed(seed)
        print("fitting..")
        root = _fit(vectors)
        print(f"fitting took {perf_counter() - start} seconds")
        print("hashing each vector..")
        hashes = _rp_hash(root, vectors, np.zeros(len(vectors), dtype=np.uint64))
        print(f"hashing took {perf_counter() - start} seconds")
        print("Sorting to build searchable hash table")
        sorted_idxs, sorted_hashes = _hash_table(hashes)
        return RandomProjectionTree(root, sorted_hashes, sorted_idxs)

    @staticmethod
    def load(path: str):
        """Load the tree from a file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def hash_of(self, vectors: np.ndarray[np.float64]) -> np.ndarray[np.uint64]:
        """Get the hash of a vector."""
        if len(vectors.shape) == 1:
            vectors = np.array([vectors])
        hashes = np.zeros(dtype=np.uint64, shape=(len(vectors)))
        return _rp_hash(self.root, vectors, hashes)

    def query(self, query: np.ndarray[np.float64]) -> np.ndarray[np.uint64]:
        """Run a query against the tree."""
        most_similar = np.searchsorted(self.sorted_hashes, self.hash_of(query))
        return self.sorted_idxs[most_similar], None

    def _depth(self, tree):
        """Get the depth of a tree."""
        if tree is None:
            return 0
        else:
            return 1 + max(self._depth(tree[1]), self._depth(tree[2]))

    def depth(self):
        """Get the depth of the tree."""
        return self._depth(self.root)

    def debug_dump(self, terms, vectors, term_id):
        """Dump a debug representation of the tree."""
        idx = np.argwhere(self.sorted_idxs == term_id)[0][0]
        num_to_dump = 10

        king = np.array([vectors[3598]])
        king_nn = np.dot(vectors, king[0]).argsort()[::-1][:10]
        king_fn = np.dot(vectors, king[0]).argsort()[:10]

        hashes_to_dump = self.sorted_hashes[idx - (num_to_dump // 2):idx + (num_to_dump // 2)]
        idxs_to_dump = self.sorted_idxs[idx - (num_to_dump // 2):idx + (num_to_dump // 2)]

        last_vector = None
        for (idx, bit_hash) in zip(idxs_to_dump, hashes_to_dump):
            dotted = np.dot(vectors[idx], last_vector) if last_vector is not None else 0
            term = terms[idx].replace("\n", "")
            print(f"{bit_hash:064b} {idx}:{term} -- gt:{dotted}")
            last_vector = vectors[idx]

        print("--------------------------------")
        # print(f" king: {self.hash_of(king)[0]:064b}")
        # print(f"queen: {self.hash_of(queen)[0]:064b}")
        for nn in king_nn:
            nn_term = terms[nn].replace("\n", "").rjust(17)
            hash_str = self.hash_of(np.array([vectors[nn]]))[0]
            hash_str = f"{hash_str:064b}"
            print(f"{nn_term}: {hash_str[-DEFAULT_DEPTH:]}")

        print("--------------------------------")
        for fn in king_fn:
            fn_term = terms[fn].replace("\n", "").rjust(17)
            hash_str = self.hash_of(np.array([vectors[fn]]))[0]
            hash_str = f"{hash_str:064b}"
            print(f"{fn_term}: {hash_str[-DEFAULT_DEPTH:]}")
        import pdb; pdb.set_trace()
