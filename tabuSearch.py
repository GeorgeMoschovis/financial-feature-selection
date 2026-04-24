import random
from collections import deque


def tabu_search_feature_selection(
    n_features: int,
    evaluate,               # callable: frozenset[int] -> float (accuracy)
    max_iter: int = 1000,
    tabu_size: int = 10,
    init_subset: set = None,
    seed: int = None,
) -> tuple[frozenset, float, list]:
    """
    Tabu search over feature subsets.

    Parameters
    ----------
    n_features  : total number of features available
    evaluate    : function that takes a frozenset of feature indices
                  and returns a float accuracy score
    max_iter    : number of iterations (paper uses 1000)
    tabu_size   : tabu list capacity (paper uses 10)
    init_subset : optional starting subset; random if None
    seed        : optional random seed for reproducibility

    Returns
    -------
    best_subset : frozenset of selected feature indices
    best_acc    : accuracy of best_subset
    history     : list of (current_acc, best_acc) per iteration
    """
    if seed is not None:
        random.seed(seed)

    all_features = set(range(n_features))

    # --- initialisation ---
    if init_subset is not None:
        current = frozenset(init_subset)
    else:
        current = frozenset(
            i for i in all_features if random.random() < 0.5
        ) or frozenset({random.randint(0, n_features - 1)})

    best = current
    best_acc = evaluate(current)
    current_acc = best_acc

    tabu_list = deque(maxlen=tabu_size)   # FIFO, auto-evicts oldest
    tabu_list.append(current)

    history = [(current_acc, best_acc)]

    # --- main loop ---
    for _ in range(max_iter):

        # generate all single-flip neighbours
        neighbors = []
        for i in all_features:
            if i in current:
                nb = current - {i}         # remove feature i
            else:
                nb = current | {i}         # add feature i
            if len(nb) > 0:
                neighbors.append(nb)

        # select best admissible neighbour
        best_nb = None
        best_nb_acc = -1.0

        for nb in neighbors:
            nb_acc = evaluate(nb)
            is_tabu = nb in tabu_list
            aspiration_met = nb_acc > best_acc

            if (not is_tabu) or aspiration_met:
                if nb_acc > best_nb_acc:
                    best_nb = nb
                    best_nb_acc = nb_acc

        # fallback: all neighbours tabu, no aspiration met
        if best_nb is None:
            best_nb = random.choice(neighbors)
            best_nb_acc = evaluate(best_nb)

        # move
        current = best_nb
        current_acc = best_nb_acc
        tabu_list.append(current)

        # update global best
        if current_acc > best_acc:
            best = current
            best_acc = current_acc

        history.append((current_acc, best_acc))

    return best, best_acc, history