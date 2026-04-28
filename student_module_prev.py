"""
student_module.py  –  STUDENT VERSION
======================================
Graph Machine Learning Lab: Shallow Node Embeddings with Random Walks

Your task is to implement every function that currently raises
NotImplementedError.  Read the docstring carefully; it describes
the expected inputs, outputs, and the algorithm to follow.

Tips
----
* Useful imports are already at the top of this file.
* You can test individual functions in the notebook before moving on.
* numpy.random.choice is your friend for sampling from a distribution.
* When stuck on the maths, re-read the relevant notebook section.
"""

import random
import numpy as np
import networkx as nx
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ===========================================================================
# SECTION 1 – DeepWalk utilities
# ===========================================================================

def uniform_random_walk(G: nx.Graph, start: int, length: int) -> list:
    """
    Perform a single uniform (unbiased) random walk on graph G.

    At each step, choose one of the current node's neighbours uniformly at
    random and append it to the walk.  Store node IDs as **strings** so that
    the walk can be fed directly into Word2Vec (every node is a "word").

    Algorithm
    ---------
    1. Initialise the walk as [str(start)].
    2. Repeat `length` times:
       a. Get the list of neighbours of the current node.
       b. If there are no neighbours, stop early.
       c. Sample one neighbour uniformly at random.
       d. Append str(sampled_neighbour) to the walk.
       e. Update the current node.
    3. Return the walk.

    Parameters
    ----------
    G      : NetworkX graph to walk on.
    start  : Starting node ID (integer).
    length : Number of steps (the walk will contain at most length+1 nodes).

    Returns
    -------
    walk : list of str, e.g. ['0', '3', '7', ...]

    Example
    -------
    >>> G = nx.erdos_renyi_graph(10, 0.4, seed=1)
    >>> walk = uniform_random_walk(G, start=0, length=5)
    >>> len(walk)  # should be 6 (start + 5 steps)
    6
    >>> all(isinstance(n, str) for n in walk)  # node IDs must be strings
    True
    """
    # ---- YOUR CODE HERE ------------------------------------------------- #
    walk = [str(start)]
    current = start
    
    for _ in range(length):
        neighbours = list(G.neighbors(current))
        if not neighbours:
            break

        next_node = np.random.choice(neighbours)
        walk.append(str(next_node))
        current = next_node
    
    return walk
    # --------------------------------------------------------------------- #


def generate_walks(G: nx.Graph, num_walks: int, walk_length: int) -> list:
    """
    Generate a corpus of random walks over **all** nodes in G.

    For each training epoch (out of `num_walks`), shuffle the node order and
    start one walk from every node.  Shuffling prevents the model from
    seeing nodes in a fixed order every epoch.

    Parameters
    ----------
    G           : NetworkX graph.
    num_walks   : Number of walks to start from each node.
    walk_length : Number of steps per walk.

    Returns
    -------
    walks : list of lists of str  (a "corpus" for Word2Vec)

    Example
    -------
    >>> G = nx.karate_club_graph()
    >>> walks = generate_walks(G, num_walks=10, walk_length=5)
    >>> len(walks) == G.number_of_nodes() * 10
    True
    """
    # ---- YOUR CODE HERE ------------------------------------------------- #
    walks = []
    nodes = list(G.nodes())

    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = uniform_random_walk(G, start=node, length=walk_length)
            walks.append(walk)
    
    return walks
    # --------------------------------------------------------------------- #


def train_embedding(
    walks: list,
    vector_size: int = 128,
    window: int = 10,
    epochs: int = 30,
    seed: int = 0,
) -> Word2Vec:
    """
    Train a skip-gram Word2Vec model on a corpus of random walks.

    Each walk is a sentence; each node ID is a word.  Use hierarchical
    softmax (hs=1) – this is what the original DeepWalk paper does.

    Parameters
    ----------
    walks       : Corpus produced by generate_walks().
    vector_size : Dimensionality of the node embedding vectors.
    window      : Context window size for skip-gram.
    epochs      : Number of training epochs.
    seed        : Random seed for reproducibility.

    Returns
    -------
    model : trained gensim Word2Vec instance.

    Hints
    -----
    * Create the model with Word2Vec(walks, hs=1, sg=1, ...).
    * Don't forget to build the vocabulary before training: model.build_vocab(walks).
    * Then call model.train(walks, total_examples=model.corpus_count,
      epochs=epochs).
    * Set workers=1 to keep results deterministic.
    """
    # ---- YOUR CODE HERE ------------------------------------------------- #
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=0,
        sg=1,
        hs=1,
        workers=1,
        seed=seed,
    )
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=epochs)
    return model
    # --------------------------------------------------------------------- #


def train_classifier(
    model: Word2Vec,
    labels: np.ndarray,
    train_mask: list,
    seed: int = 0,
) -> RandomForestClassifier:
    """
    Fit a Random Forest classifier on node embeddings.

    Use model.wv[train_mask] to retrieve the embedding matrix for the
    training nodes.

    Important: `train_mask` must contain node IDs as **strings** (e.g.
    ['0', '2', '4']), because walks store node IDs as text.  Use
    [int(n) for n in train_mask] to convert back to integers when
    indexing into the `labels` array.

    Parameters
    ----------
    model      : Trained Word2Vec model.
    labels     : Array of ground-truth node labels (integers).
    train_mask : List of node IDs as strings.
    seed       : Random seed for the classifier.

    Returns
    -------
    clf : Fitted RandomForestClassifier.
    """
    # ---- YOUR CODE HERE ------------------------------------------------- #
    X_train = model.wv[train_mask]
    y_train = labels[[int(n) for n in train_mask]]

    clf = RandomForestClassifier(random_state=seed)
    clf.fit(X_train, y_train)
    return clf
    # --------------------------------------------------------------------- #


def evaluate_classifier(
    clf: RandomForestClassifier,
    model: Word2Vec,
    labels: np.ndarray,
    test_mask: list,
) -> float:
    """
    Evaluate a fitted classifier on held-out nodes.

    Parameters
    ----------
    clf       : Fitted classifier.
    model     : Trained Word2Vec model.
    labels    : Full array of ground-truth node labels.
    test_mask : List of node IDs as **strings**.

    Returns
    -------
    acc : Float accuracy in [0, 1].
    """
    # ---- YOUR CODE HERE ------------------------------------------------- #
    X_test = model.wv[test_mask]
    y_test = labels[[int(n) for n in test_mask]]

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
    # --------------------------------------------------------------------- #


# ===========================================================================
# SECTION 2 – Node2Vec utilities (biased random walks)
# ===========================================================================

def biased_next_node(
    G: nx.Graph,
    previous,          # int or None
    current: int,
    p: float,
    q: float,
) -> int:
    """
    Sample the next node for a Node2Vec walk.

    Node2Vec defines a **second-order** random walk: the transition
    probability from `current` to each neighbour depends on the distance
    between that neighbour and the **previous** node.

    For each neighbour `v` of `current`, compute an unnormalised weight
    alpha(v) according to:

        alpha(v) = 1/p   if v == previous            (return step)
        alpha(v) = 1     if (v, previous) is an edge  (same distance)
        alpha(v) = 1/q   otherwise                    (explore farther)

    Then normalise the alphas to get a valid probability distribution and
    sample one neighbour.

    When `previous` is None (first step), fall back to a **uniform**
    transition (no bias yet).

    Parameters
    ----------
    G        : NetworkX graph.
    previous : Previously visited node (int), or None for the first step.
    current  : Current node (int).
    p        : Return parameter.  High p  -> less likely to backtrack.
    q        : In-out parameter.  Low q  -> BFS-like (local).
                                  High q -> DFS-like (exploratory).

    Returns
    -------
    next_node : int, the sampled next node.
    """
    neighbours = list(G.neighbors(current))

    if previous is None:
        # ---- YOUR CODE HERE (first step, uniform transition) ------------ #
        return np.random.choice(neighbours)
        # ----------------------------------------------------------------- #

    # ---- YOUR CODE HERE (biased transition) ----------------------------- #
    weights = []
    for v in neighbours:
        if v == previous:
            # Return to previous node
            weights.append(1.0 / p)
        elif G.has_edge(v, previous):
            # Shared neighbour (distance 1)
            weights.append(1.0)
        else:
            # Explore farther (distance 2+)
            weights.append(1.0 / q)

    weights = np.array(weights)
    probs = weights / weights.sum()

    return np.random.choice(neighbours, p=probs)
    # --------------------------------------------------------------------- #


def biased_random_walk(
    G: nx.Graph,
    start: int,
    length: int,
    p: float = 1.0,
    q: float = 1.0,
) -> list:
    """
    Perform a single second-order biased random walk (Node2Vec).

    At each step, call biased_next_node to decide where to go next,
    making sure to pass the correct `previous` node.

    Parameters
    ----------
    G      : NetworkX graph.
    start  : Starting node (int).
    length : Number of steps.
    p      : Return parameter.
    q      : In-out parameter.

    Returns
    -------
    walk : list of str node IDs.
    """
    # ---- YOUR CODE HERE ------------------------------------------------- #
    walk = [str(start)]

    current = start
    previous = None
    
    for _ in range(length):
        next_node = biased_next_node(G, previous, current, p, q)
        walk.append(str(next_node))
        previous = current
        current = next_node
    
    return walk
    # --------------------------------------------------------------------- #


def generate_biased_walks(
    G: nx.Graph,
    num_walks: int,
    walk_length: int,
    p: float = 1.0,
    q: float = 1.0,
) -> list:
    """
    Generate a corpus of Node2Vec biased random walks over all nodes in G.

    Same shuffling logic as generate_walks, but call biased_random_walk
    instead of uniform_random_walk.

    Parameters
    ----------
    G           : NetworkX graph.
    num_walks   : Walks per node.
    walk_length : Steps per walk.
    p           : Return parameter.
    q           : In-out parameter.

    Returns
    -------
    walks : list of lists of str.
    """
    # ---- YOUR CODE HERE ------------------------------------------------- #
    walks = []
    nodes = list(G.nodes())

    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = biased_random_walk(G, start=node, length=walk_length, p=p, q=q)
            walks.append(walk)
    
    return walks
    # --------------------------------------------------------------------- #
