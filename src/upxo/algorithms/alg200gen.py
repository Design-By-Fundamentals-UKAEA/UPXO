from copy import deepcopy
from random import sample as sample_rand
import numpy.random as rand
from upxo.pxtal.mcgs2_temporal_slice import mcgs2_grain_structure as GS2d


def run(uisim, uiint, uidata, uigrid,
        xgr, ygr, zgr, px_size,
        _a, _b, _c, S, AIA0, AIA1, display_messages):
    # ---------------------------------------------
    print("Using ALG-200gen: SA's NL-1 weighted Q-Pott's model:")
    print('|' + 15*'-'+' MC SIM RUN IN PROGRESS on: ALG200gen' + 15*'-' + '|')
    gs = {}
    # Build the Non-Locality Matrix
    NLM_00, NLM_01, NLM_02 = _a  # Unpack 3 colms of 1st row
    NLM_10, NLM_11, NLM_12 = _b  # Unpack 3 colms of 2nd row
    NLM_20, NLM_21, NLM_22 = _c  # Unpack 3 colms of 3rd row
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
            break
        else:
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
                    Neigh = [ssub_00, ssub_01, ssub_02,
                             ssub_10, ssub_11, ssub_12,
                             ssub_20, ssub_21, ssub_22]
                    if min(Neigh) != max(Neigh):
                        DelH1 = NLM_00*int(ssub_11 == ssub_00) + \
                                NLM_01*int(ssub_11 == ssub_01) + \
                                NLM_02*int(ssub_11 == ssub_02) + \
                                NLM_10*int(ssub_11 == ssub_10) + \
                                NLM_12*int(ssub_11 == ssub_12) + \
                                NLM_20*int(ssub_11 == ssub_20) + \
                                NLM_21*int(ssub_11 == ssub_21) + \
                                NLM_22*int(ssub_11 == ssub_22)
                        # ---------------------------------------------
                        # If the sampling is to be selected without
                        # weightage to dominant neighbour state, then:
                        Neigh = set([x for x in Neigh if x != ssub_11])
                        # ---------------------------------------------
                        ssub_11_b = sample_rand(Neigh, 1)[0]
                        DelH2 = NLM_00*int(ssub_11_b == ssub_00) + \
                                NLM_01*int(ssub_11_b == ssub_01) + \
                                NLM_02*int(ssub_11_b == ssub_02) + \
                                NLM_10*int(ssub_11_b == ssub_10) + \
                                NLM_12*int(ssub_11_b == ssub_12) + \
                                NLM_20*int(ssub_11_b == ssub_20) + \
                                NLM_21*int(ssub_11_b == ssub_21) + \
                                NLM_22*int(ssub_11_b == ssub_22)
                        if DelH2 >= DelH1:
                            S[s0, s1] = ssub_11_b
                        elif uisim.consider_boltzmann_probability:
                            if uisim.s_boltz_prob[int(ssub_11_b-1)] < rand.random():
                                S[s0, s1] = ssub_11_b
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
    print('|' + 15*'-'+' MC SIM RUN COMPLETED on: ALG200gen' + 15*'-' + '|')
    fully_annealed = {'fully_annealed': fully_annealed,
                      'm': fully_annealed_at_m}
    return gs, fully_annealed
