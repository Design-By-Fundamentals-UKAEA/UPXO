def mc_iterations_3d_alg224(self):
    """
    DESIGNED TO ACHIEVE: Bi-modal grain size distribution

    Each of the initial set of iterations is to contain the following

    STEP 1: Do the regular iteration using any of the 200 series of
    algorithms

    STEP 2: Identify grains and their neighbours

    STEP 3: Calculate state partitioned grain area distribution

    STEP 4: Identify small grains with areas less than P % of mean area
    for each state

    STEP 5: Select the state with the largest mean area: S_large

    STEP 6: Select the state with the smallest mean area: S_small

    STEP 7: Prepare a merger list comprising of two columns. First column
    is to have the global grain IDs of certain grains belonging to S_small.
    The second column is to have a list of global grain IDs of neighbouring
    grains belonging to S_large. If for a grain of S_small, no neighbouring
    grains of S_large exit, then cancel the merger operation for the
    current S_small grain. Iterate through all the remaining grains.

    STEP 8: Calculate the grain area distribution. Calculate the modality
    Calculate the shift in peaks.

    STEP 9: If the peak shift is in the direction of target peak, then
    accept the present iteration using a iteration transition probability.
    """
    pass