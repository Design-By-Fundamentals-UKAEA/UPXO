def mc_iterations_3d_alg230(self):
    """
    230 SERIES OF ALGORITHMS

    This series belongs to the Cluster Monte-Carlo Algorithms. Some of them
    may be well known existing ones, some of them, developed by the lead
    developer, Dr. Sunil Anandatheertha.
    ------------------------------------------------------------
    ALGORITHM 230
    ------------------------------------------------------------
    DESIGNED TO ACHIEVE: Multi-modal grain structure
    DEVELOPED BY: Dr. Sunil Anandatheertha
    ------------------------------------------------------------
    Each of the initial set of iterations is to contain the following

    STEP 1: Do the regular iteration using any of the 200 series of
    algorithms

    STEP 2: Identify grains and their neighbours

    STEP 3: Identify a state at random: S1

    STEP 4: Identify all grains of S1. Build a single list having
    IDs of all grains which neighbour the grains of S1.

    STEP 5: Identify the most frequent state amongst these grains, which
    would be S1_neigh_mostfrequent

    STEP 6: Flip the states of all S1 grains to S1_neigh_mostfrequent

    STEP 7: Characterise the grain structure.
    """
    pass