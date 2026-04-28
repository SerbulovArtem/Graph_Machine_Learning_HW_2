"""
student_module.py  –  STUDENT VERSION
=========================================
Graph Machine Learning Lab: Node Features and Graph Neural Networks

Your task is to implement every function and method that currently raises
NotImplementedError.  Read the docstring carefully; it describes the
expected inputs, outputs, and the algorithm to follow.

Tips
----
* Useful imports are already at the top.
* Test each task in the notebook before moving on to the next one.
* When stuck, re-read the relevant notebook section.
* For PyTorch tasks: remember that torch.nn.functional is imported as F.
"""

import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Shared utility (used in all three approaches)
# ---------------------------------------------------------------------------

def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Return the fraction of correct predictions (provided – do not modify)."""
    return (y_pred == y_true).float().mean().item()


# ===========================================================================
# SECTION 1 – Shallow embedding utilities  (Task 1)
# ===========================================================================

def get_embedding_matrix(wv, node_ids: list) -> np.ndarray:
    """
    Extract a numpy embedding matrix from a trained Word2Vec model.

    Parameters
    ----------
    wv       : KeyedVectors object (model.wv from a trained gensim Word2Vec).
    node_ids : List of node IDs as **strings**, e.g. ['0', '1', '2', ...].
               These must match the keys that were used when training the model
               (remember: walks store node IDs as text).

    Returns
    -------
    X : np.ndarray of shape (len(node_ids), embedding_dim)
        Row i is the embedding vector of node_ids[i].

    Example
    -------
    >>> X = get_embedding_matrix(model.wv, ['0', '1', '2'])
    >>> X.shape
    (3, 128)
    >>> isinstance(X, np.ndarray)
    True
    """
    # ---- YOUR CODE HERE -------------------------------------------------- #
    return np.array([wv[node_id] for node_id in node_ids])
    # ---------------------------------------------------------------------- #


# ===========================================================================
# SECTION 2 – MLP: features only  (Tasks 2 and 3)
# ===========================================================================

class MLP(torch.nn.Module):
    """Two-layer Multilayer Perceptron for node classification.

    Architecture
    ------------
    Linear(dim_in, dim_h) → ReLU → Linear(dim_h, dim_out) → log-softmax

    Parameters
    ----------
    dim_in  : int – number of input features per node.
    dim_h   : int – hidden layer width.
    dim_out : int – number of output classes.
    """

    def __init__(self, dim_in: int, dim_h: int, dim_out: int):
        super().__init__()
        self.linear1 = Linear(dim_in, dim_h)
        self.linear2 = Linear(dim_h, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class log-probabilities for every node.

        Algorithm
        ---------
        1. Apply self.linear1 to x.
        2. Apply ReLU non-linearity.
        3. Apply self.linear2.
        4. Return F.log_softmax(result, dim=1).

        Parameters
        ----------
        x : FloatTensor of shape (num_nodes, dim_in)

        Returns
        -------
        out : FloatTensor of shape (num_nodes, dim_out)
              Log-probabilities over classes.
        """
        # ---- YOUR CODE HERE ---------------------------------------------- #
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)
        # ------------------------------------------------------------------ #

    # -----------------------------------------------------------------------
    # Provided – do not modify
    # -----------------------------------------------------------------------

    def fit(self, data, epochs: int = 100, lr: float = 0.01):
        """Train with Adam + cross-entropy on the training mask."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                train_acc = accuracy(
                    out[data.train_mask].argmax(dim=1), data.y[data.train_mask]
                )
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(
                    out[data.val_mask].argmax(dim=1), data.y[data.val_mask]
                )
                print(
                    f"Epoch {epoch:>3} | Loss: {loss:.3f} | "
                    f"Train Acc: {train_acc * 100:>5.2f}% | "
                    f"Val Loss: {val_loss:.2f} | "
                    f"Val Acc: {val_acc * 100:.2f}%"
                )

    @torch.no_grad()
    def test(self, data) -> float:
        """Return accuracy on the test mask."""
        self.eval()
        out = self(data.x)
        return accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])

    @torch.no_grad()
    def get_hidden(self, data) -> torch.Tensor:
        """Return intermediate representations after the first layer (for t-SNE)."""
        self.eval()
        h = torch.relu(self.linear1(data.x))
        return h


# ===========================================================================
# SECTION 3 – Vanilla GNN: graph + features  (Tasks 4, 5, and 6)
# ===========================================================================

def build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Build a dense adjacency matrix with self-loops from a PyG edge_index.

    In a Vanilla GNN the node representation at layer k+1 is computed as:

        H^(k+1) = A_hat * H^(k) * W^(k)

    where A_hat = A + I  (adjacency with self-loops, so each node also
    aggregates its own previous representation).

    Algorithm
    ---------
    1. Create a zero FloatTensor of shape (num_nodes, num_nodes).
    2. Scatter ones at positions given by edge_index:
         A[edge_index[0], edge_index[1]] = 1
    3. Add the identity matrix (self-loops):
         A = A + torch.eye(num_nodes)
    4. Return A  (values in {0, 1, 2} for unweighted graphs; this is fine
       for a Vanilla GNN, though GCN would normalise further).

    Parameters
    ----------
    edge_index : LongTensor of shape (2, E)  – PyG edge index.
    num_nodes  : int – total number of nodes.

    Returns
    -------
    A : FloatTensor of shape (num_nodes, num_nodes)

    Example
    -------
    >>> import torch
    >>> ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    >>> A = build_adjacency(ei, 3)
    >>> A.shape
    torch.Size([3, 3])
    >>> A.diagonal().tolist()   # all ones (self-loops)
    [1.0, 1.0, 1.0]
    """
    # ---- YOUR CODE HERE -------------------------------------------------- #
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    A[edge_index[0], edge_index[1]] = 1.0
    A = A + torch.eye(num_nodes)
    return A
    # ---------------------------------------------------------------------- #


def build_adjacency_normalized(
    edge_index: torch.Tensor,
    num_nodes: int,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Build a dense adjacency matrix with self-loops, optionally row-normalised.

    When normalize=False this is a drop-in replacement for build_adjacency:
    it returns A + I, the plain adjacency with self-loops.

    When normalize=True the aggregation becomes a weighted average instead of
    a sum.  Each row is divided by its degree in the augmented graph:

        A_norm = D^{-1} * (A + I)

    where D_{ii} = sum_j (A + I)_{ij}  (number of neighbours + self).

    Effect: a hub node with 100 neighbours and an isolated node with 0
    neighbours both receive representations on the same scale, because each
    neighbour contributes 1/degree of its vector instead of a full copy.

    Parameters
    ----------
    edge_index : LongTensor of shape (2, E)  – PyG edge index.
    num_nodes  : int – total number of nodes.
    normalize  : bool – if True apply row-normalisation; default False.

    Returns
    -------
    A : FloatTensor of shape (num_nodes, num_nodes)

    Example
    -------
    >>> ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    >>> A = build_adjacency_normalized(ei, 2, normalize=True)
    >>> A.sum(dim=1).tolist()      # each row sums to 1.0
    [1.0, 1.0]
    >>> A_plain = build_adjacency_normalized(ei, 2, normalize=False)
    >>> A_plain.diagonal().tolist()  # self-loops present
    [1.0, 1.0]

    Algorithm
    ---------
    1. Build the plain adjacency with self-loops  (same as build_adjacency).
    2. If normalize is False, return immediately.
    3. Compute row degrees: deg = A.sum(dim=1, keepdim=True)  →  shape (N, 1).
       Guard isolated nodes: replace any zero entry with 1.0.
    4. Divide: A = A / deg.
    5. Return A.
    """
    # ---- YOUR CODE HERE -------------------------------------------------- #
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    A[edge_index[0], edge_index[1]] = 1.0
    A = A + torch.eye(num_nodes)

    if not normalize:
        return A
    
    deg = A.sum(dim=1, keepdim=True)

    A = A / deg

    return A
    # ---------------------------------------------------------------------- #


class VanillaGNNLayer(torch.nn.Module):
    """A single Vanilla GNN layer.

    Computes: output = A_hat * (x * W)
    i.e. first a linear (weight) transformation, then neighbourhood
    aggregation via matrix multiplication with the adjacency.

    Parameters
    ----------
    dim_in  : int – input feature dimension.
    dim_out : int – output embedding dimension.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear = Linear(dim_in, dim_out, bias=False)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation then aggregate from neighbours.

        Algorithm
        ---------
        1. Apply self.linear to x  →  shape (num_nodes, dim_out).
        2. Left-multiply by adjacency:  adjacency @ result
           (this aggregates each node's transformed features with its
           neighbours' transformed features, including itself via self-loops).
        3. Return the result.

        Parameters
        ----------
        x         : FloatTensor of shape (num_nodes, dim_in)
        adjacency : FloatTensor of shape (num_nodes, num_nodes)

        Returns
        -------
        out : FloatTensor of shape (num_nodes, dim_out)
        """
        # ---- YOUR CODE HERE ---------------------------------------------- #
        x = self.linear(x)
        x = adjacency @ x
        return x
        # ------------------------------------------------------------------ #


class VanillaGNN(torch.nn.Module):
    """Two-layer Vanilla Graph Neural Network for node classification.

    Architecture
    ------------
    VanillaGNNLayer(dim_in, dim_h) → ReLU →
    VanillaGNNLayer(dim_h, dim_out) → log-softmax

    Parameters
    ----------
    dim_in  : int – input feature dimension.
    dim_h   : int – hidden layer width.
    dim_out : int – number of output classes.
    """

    def __init__(self, dim_in: int, dim_h: int, dim_out: int):
        super().__init__()
        self.gnn1 = VanillaGNNLayer(dim_in, dim_h)
        self.gnn2 = VanillaGNNLayer(dim_h, dim_out)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Compute class log-probabilities for every node.

        Algorithm
        ---------
        1. Pass (x, adjacency) through self.gnn1.
        2. Apply ReLU.
        3. Pass result and adjacency through self.gnn2.
        4. Return F.log_softmax(result, dim=1).

        Parameters
        ----------
        x         : FloatTensor of shape (num_nodes, dim_in)
        adjacency : FloatTensor of shape (num_nodes, num_nodes)

        Returns
        -------
        out : FloatTensor of shape (num_nodes, dim_out)
        """
        # ---- YOUR CODE HERE ---------------------------------------------- #
        x = self.gnn1(x, adjacency)
        x = F.relu(x)
        x = self.gnn2(x, adjacency)
        return F.log_softmax(x, dim=1)
        # ------------------------------------------------------------------ #

    # -----------------------------------------------------------------------
    # Provided – do not modify
    # -----------------------------------------------------------------------

    def fit(self, data, adjacency: torch.Tensor, epochs: int = 100, lr: float = 0.01):
        """Train with Adam + cross-entropy on the training mask."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, adjacency)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                train_acc = accuracy(
                    out[data.train_mask].argmax(dim=1), data.y[data.train_mask]
                )
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(
                    out[data.val_mask].argmax(dim=1), data.y[data.val_mask]
                )
                print(
                    f"Epoch {epoch:>3} | Loss: {loss:.3f} | "
                    f"Train Acc: {train_acc * 100:>5.2f}% | "
                    f"Val Loss: {val_loss:.2f} | "
                    f"Val Acc: {val_acc * 100:.2f}%"
                )

    @torch.no_grad()
    def test(self, data, adjacency: torch.Tensor) -> float:
        """Return accuracy on the test mask."""
        self.eval()
        out = self(data.x, adjacency)
        return accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])

    @torch.no_grad()
    def get_hidden(self, data, adjacency: torch.Tensor) -> torch.Tensor:
        """Return layer-1 embeddings (for t-SNE)."""
        self.eval()
        h = torch.relu(self.gnn1(data.x, adjacency))
        return h
