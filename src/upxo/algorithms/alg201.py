from copy import deepcopy
from random import sample as sample_rand
import numpy as np
import numpy.random as rand
from upxo.pxtal.mcgs2_temporal_slice import mcgs2_grain_structure as GS2d
import random
from numba import njit

@njit
def unique_with_counts(arr):
    """
    This is because, current numba is incompatible with the return_counts
    argument in np.unique.
    """
    # Sort the array for grouping identical values
    sorted_arr = np.sort(arr)
    unique_vals = []
    counts = []

    # Initialize the first unique value and count
    if len(sorted_arr) > 0:
        current_val = sorted_arr[0]
        count = 1

        for i in range(1, len(sorted_arr)):
            if sorted_arr[i] == current_val:
                count += 1
            else:
                unique_vals.append(current_val)
                counts.append(count)
                current_val = sorted_arr[i]
                count = 1

        # Append the last value and count
        unique_vals.append(current_val)
        counts.append(count)

    return np.array(unique_vals), np.array(counts)

@njit
def mcloop_alg201(cbp, sbp, S, AIA0, AIA1, NLM, rsfso):
    # rsfso = 2
    for s0 in list(range(S.shape[0])):  # along axis 0
        s00, s01, s02 = s0+0, s0+1, s0+2
        for s1 in list(range(S.shape[1])):  # along axis 1
            s10, s11, s12 = s1+0, s1+1, s1+2
            ssub_00 = S[AIA0[s00, s10], AIA1[s00, s10]]
            ssub_01 = S[AIA0[s01, s10], AIA1[s01, s10]]
            ssub_02 = S[AIA0[s02, s10], AIA1[s02, s10]]
            ssub_10 = S[AIA0[s00, s11], AIA1[s00, s11]]
            ssub_11 = S[AIA0[s01, s11], AIA1[s01, s11]]
            ssub_12 = S[AIA0[s02, s11], AIA1[s02, s11]]
            ssub_20 = S[AIA0[s00, s12], AIA1[s00, s12]]
            ssub_21 = S[AIA0[s01, s12], AIA1[s01, s12]]
            ssub_22 = S[AIA0[s02, s12], AIA1[s02, s12]]
            Neigh = np.array([ssub_00, ssub_01, ssub_02,
                              ssub_10, ssub_11, ssub_12,
                              ssub_20, ssub_21, ssub_22])
            if Neigh.min() != Neigh.max():
                DelH1 = NLM[0, 0]*int(ssub_11 == ssub_00) + \
                        NLM[0, 1]*int(ssub_11 == ssub_01) + \
                        NLM[0, 2]*int(ssub_11 == ssub_02) + \
                        NLM[1, 0]*int(ssub_11 == ssub_10) + \
                        NLM[1, 2]*int(ssub_11 == ssub_12) + \
                        NLM[2, 0]*int(ssub_11 == ssub_20) + \
                        NLM[2, 1]*int(ssub_11 == ssub_21) + \
                        NLM[2, 2]*int(ssub_11 == ssub_22)
                # ---------------------------------------------
                # If the sampling is to be selected with
                # weightage to dominant neighbour state, then:
                Neigh = Neigh[Neigh != ssub_11]
                # ---------------------------------------------
                # Neigh_, counts_ = np.unique(Neigh, return_counts=True)
                Neigh_, counts_ = unique_with_counts(Neigh)

                counts_ = np.argsort(counts_)[::-1]
                counts_ = counts_[:rsfso]

                random_index = np.random.randint(0, len(counts_))
                ssub_11_b = Neigh_[random_index]
                # ---------------------------------------------
                DelH2 = NLM[0, 0]*int(ssub_11_b == ssub_00) + \
                        NLM[0, 1]*int(ssub_11_b == ssub_01) + \
                        NLM[0, 2]*int(ssub_11_b == ssub_02) + \
                        NLM[1, 0]*int(ssub_11_b == ssub_10) + \
                        NLM[1, 2]*int(ssub_11_b == ssub_12) + \
                        NLM[2, 0]*int(ssub_11_b == ssub_20) + \
                        NLM[2, 1]*int(ssub_11_b == ssub_21) + \
                        NLM[2, 2]*int(ssub_11_b == ssub_22)
                if DelH2 >= DelH1:
                    S[s0, s1] = ssub_11_b
                elif cbp:
                    if sbp[int(ssub_11_b-1)] < np.random.random():
                        S[s0, s1] = ssub_11_b
    return S

def run(uisim, uiint, uidata, uigrid, rsfso, xgr, ygr, zgr, px_size,
        _a, _b, _c, S, AIA0, AIA1, display_messages):
    # ---------------------------------------------
    print("Using ALG-200: SA's NL-1 weighted Q-Pott's model:")
    print('|' + 15*'-'+' MC SIM RUN IN PROGRESS on: ALG201' + 15*'-' + '|')
    gs = {}
    # Build the Non-Locality Matrix
    NLM_00, NLM_01, NLM_02 = _a  # Unpack 3 colms of 1st row
    NLM_10, NLM_11, NLM_12 = _b  # Unpack 3 colms of 2nd row
    NLM_20, NLM_21, NLM_22 = _c  # Unpack 3 colms of 3rd row
    NLM = np.array([[NLM_00, NLM_01, NLM_02],
                    [NLM_10, NLM_11, NLM_12],
                    [NLM_20, NLM_21, NLM_22]
                    ])
    print('============================================')
    print(NLM)
    print('============================================')
    # ---------------------------------------------
    # Begin modified Markov-Chain annealing iterations
    fully_annealed = False
    fully_annealed_at_m = None
    for m in range(uisim.mcsteps):
        if S.min() == S.max():
            print(30*'.')
            print(f'Single crystal achieved at iteration {m}.')
            fully_annealed, fully_annealed_at_m = True, m
            # Store the last temporal slice as a UPXO grain structure by
            # default
            gs[m] = GS2d(m=m,
                         dim=uigrid.dim,
                         uidata=uidata,
                         px_size=px_size,
                         S_total=uisim.S,
                         xgr=xgr,
                         ygr=ygr,
                         uigrid=uigrid,
                         )
            gs[m].s = deepcopy(S)
            if display_messages:
                print(f"GS temporal slice {m} stored\n\n"
                      "!! MONTE-CARLO ALG.202 run ended !!\n")
                # 3952493
            break
        else:
            cbp = uisim.consider_boltzmann_probability
            sbp = uisim.s_boltz_prob
            S = mcloop_alg201(cbp, sbp, S, AIA0, AIA1, NLM, rsfso)
        cond_1 = m % uiint.mcint_save_at_mcstep_interval == 0.0
        save_msg = False
        if m==0 or cond_1 or fully_annealed:
            gs[m] = GS2d(m=m,
                         dim=uigrid.dim,
                         uidata=uidata,
                         px_size=px_size,
                         S_total=uisim.S,
                         xgr=xgr,
                         ygr=ygr,
                         uigrid=uigrid,
                         )
            gs[m].s = deepcopy(S)
            save_msg = True
            if display_messages:
                print(f"GS temporal slice {m} stored")
        if m % uiint.mcint_promt_display == 0:
            if display_messages:
                if not save_msg:
                    print(f"Monte-Carlo temporal step = {m}")
    print('|' + 15*'-'+' MC SIM RUN COMPLETED on: ALG201' + 15*'-' + '|')
    fully_annealed = {'fully_annealed': fully_annealed,
                      'm': fully_annealed_at_m}
    return gs, fully_annealed
