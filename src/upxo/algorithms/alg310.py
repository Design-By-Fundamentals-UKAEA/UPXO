from copy import deepcopy
from random import sample as sample_rand
import numpy.random as rand
from upxo.pxtal.mcgs3_temporal_slice import mcgs3_grain_structure as GS3d
from numba import njit
import numpy as np

@njit
def mcloop_alg301(cbp, sbp, S, xinda, yinda, zinda):
    xi, yi, zi = xinda, yinda, zinda
    S_sz0, S_sz1, S_sz2 = S.shape[0], S.shape[1], S.shape[2]
    for P in range(S_sz2):  # along axis 2, along plane
        for R in range(S_sz0):  # along axis 1, along row
            for C in range(S_sz1):  # along axis 0, along column.
                P0, R0, C0 = P, R, C
                P1, R1, C1 = P+1, R+1, C+1
                P2, R2, C2 = P+2, R+2, C+2

                ssub_000 = S[zi[P0, R0, C0], yi[P0, R0, C0], xi[P0, R0, C0]]
                ssub_001 = S[zi[P0, R0, C1], yi[P0, R0, C1], xi[P0, R0, C1]]
                ssub_002 = S[zi[P0, R0, C2], yi[P0, R0, C2], xi[P0, R0, C2]]
                ssub_010 = S[zi[P0, R1, C0], yi[P0, R1, C0], xi[P0, R1, C0]]
                ssub_011 = S[zi[P0, R1, C1], yi[P0, R1, C1], xi[P0, R1, C1]]
                ssub_012 = S[zi[P0, R1, C2], yi[P0, R1, C2], xi[P0, R1, C2]]
                ssub_020 = S[zi[P0, R2, C0], yi[P0, R2, C0], xi[P0, R2, C0]]
                ssub_021 = S[zi[P0, R2, C1], yi[P0, R2, C1], xi[P0, R2, C1]]
                ssub_022 = S[zi[P0, R2, C2], yi[P0, R2, C2], xi[P0, R2, C2]]

                ssub_100 = S[zi[P1, R0, C0], yi[P1, R0, C0], xi[P1, R0, C0]]
                ssub_101 = S[zi[P1, R0, C1], yi[P1, R0, C1], xi[P1, R0, C1]]
                ssub_102 = S[zi[P1, R0, C2], yi[P1, R0, C2], xi[P1, R0, C2]]
                ssub_110 = S[zi[P1, R1, C0], yi[P1, R1, C0], xi[P1, R1, C0]]
                ssub_111_a = S[zi[P1, R1, C1], yi[P1, R1, C1], xi[P1, R1, C1]]
                ssub_112 = S[zi[P1, R1, C2], yi[P1, R1, C2], xi[P1, R1, C2]]
                ssub_120 = S[zi[P1, R2, C0], yi[P1, R2, C0], xi[P1, R2, C0]]
                ssub_121 = S[zi[P1, R2, C1], yi[P1, R2, C1], xi[P1, R2, C1]]
                ssub_122 = S[zi[P1, R2, C2], yi[P1, R2, C2], xi[P1, R2, C2]]

                ssub_200 = S[zi[P2, R0, C0], yi[P2, R0, C0], xi[P2, R0, C0]]
                ssub_201 = S[zi[P2, R0, C1], yi[P2, R0, C1], xi[P2, R0, C1]]
                ssub_202 = S[zi[P2, R0, C2], yi[P2, R0, C2], xi[P2, R0, C2]]
                ssub_210 = S[zi[P2, R1, C0], yi[P2, R1, C0], xi[P2, R1, C0]]
                ssub_211 = S[zi[P2, R1, C1], yi[P2, R1, C1], xi[P2, R1, C1]]
                ssub_212 = S[zi[P2, R1, C2], yi[P2, R1, C2], xi[P2, R1, C2]]
                ssub_220 = S[zi[P2, R2, C0], yi[P2, R2, C0], xi[P2, R2, C0]]
                ssub_221 = S[zi[P2, R2, C1], yi[P2, R2, C1], xi[P2, R2, C1]]
                ssub_222 = S[zi[P2, R2, C2], yi[P2, R2, C2], xi[P2, R2, C2]]

                Neigh = np.array([ssub_000, ssub_001, ssub_002,
                                  ssub_010, ssub_011, ssub_012,
                                  ssub_020, ssub_021, ssub_022,
                                  ssub_100, ssub_101, ssub_102,
                                  ssub_110, ssub_111_a, ssub_112,
                                  ssub_120, ssub_121, ssub_122,
                                  ssub_200, ssub_201, ssub_202,
                                  ssub_210, ssub_211, ssub_212,
                                  ssub_220, ssub_221, ssub_222])

                if Neigh.min() != Neigh.max():
                    DelH1 = int(ssub_111_a == ssub_000) + \
                            int(ssub_111_a == ssub_001) + \
                            int(ssub_111_a == ssub_002) + \
                            int(ssub_111_a == ssub_010) + \
                            int(ssub_111_a == ssub_011) + \
                            int(ssub_111_a == ssub_012) + \
                            int(ssub_111_a == ssub_020) + \
                            int(ssub_111_a == ssub_021) + \
                            int(ssub_111_a == ssub_022) + \
                            int(ssub_111_a == ssub_100) + \
                            int(ssub_111_a == ssub_101) + \
                            int(ssub_111_a == ssub_102) + \
                            int(ssub_111_a == ssub_110) + \
                            int(ssub_111_a == ssub_112) + \
                            int(ssub_111_a == ssub_120) + \
                            int(ssub_111_a == ssub_121) + \
                            int(ssub_111_a == ssub_122) + \
                            int(ssub_111_a == ssub_200) + \
                            int(ssub_111_a == ssub_201) + \
                            int(ssub_111_a == ssub_202) + \
                            int(ssub_111_a == ssub_210) + \
                            int(ssub_111_a == ssub_211) + \
                            int(ssub_111_a == ssub_212) + \
                            int(ssub_111_a == ssub_220) + \
                            int(ssub_111_a == ssub_221) + \
                            int(ssub_111_a == ssub_222)
                    # ---------------------------------------------
                    Neigh = Neigh[Neigh != ssub_111_a]
                    # ---------------------------------------------
                    ssub_111_b = np.random.choice(Neigh, 1)[0]
                    # ---------------------------------------------
                    DelH2 = int(ssub_111_b == ssub_000) + \
                        int(ssub_111_b == ssub_001) + \
                        int(ssub_111_b == ssub_002) + \
                        int(ssub_111_b == ssub_010) + \
                        int(ssub_111_b == ssub_011) + \
                        int(ssub_111_b == ssub_012) + \
                        int(ssub_111_b == ssub_020) + \
                        int(ssub_111_b == ssub_021) + \
                        int(ssub_111_b == ssub_022) + \
                        int(ssub_111_b == ssub_100) + \
                        int(ssub_111_b == ssub_101) + \
                        int(ssub_111_b == ssub_102) + \
                        int(ssub_111_b == ssub_110) + \
                        int(ssub_111_b == ssub_112) + \
                        int(ssub_111_b == ssub_120) + \
                        int(ssub_111_b == ssub_121) + \
                        int(ssub_111_b == ssub_122) + \
                        int(ssub_111_b == ssub_200) + \
                        int(ssub_111_b == ssub_201) + \
                        int(ssub_111_b == ssub_202) + \
                        int(ssub_111_b == ssub_210) + \
                        int(ssub_111_b == ssub_211) + \
                        int(ssub_111_b == ssub_212) + \
                        int(ssub_111_b == ssub_220) + \
                        int(ssub_111_b == ssub_221) + \
                        int(ssub_111_b == ssub_222)
                    if DelH2 >= DelH1:
                        _p = zinda[P1, R1, C1]
                        _r = yinda[P1, R1, C1]
                        _c = xinda[P1, R1, C1]
                        S[_p, _r, _c] = ssub_111_b
                    elif cbp:
                        if sbp[int(ssub_111_b-1)] < np.random.random():
                            S[P, R, C] = ssub_111_b
    return S


def mc_iterations_3d_alg301(S=None,
                            vox_size=None, xinda=None, yinda=None, zinda=None,
                            uidata=None, uigrid=None, uisim=None, uiint=None,
                            uimesh=None, verbose=False):
    S_sz0, S_sz1, S_sz2 = S.shape[0], S.shape[1], S.shape[2]
    S_sz0_list = list(range(S_sz0))
    S_sz1_list = list(range(S_sz1))
    S_sz2_list = list(range(S_sz2))
    # --------------------------------------
    fully_annealed = False
    fully_annealed_at_m = None
    gs = {}
    print('Initiating grain growth')
    print(40*'-')
    for m in range(uisim.mcsteps):
        if S.min() == S.max():
            print(30*'.')
            print(f'Single crystal achieved at iteration {m}.')
            fully_annealed, fully_annealed_at_m = True, m
            # Store the last temporal slice as a UPXO grain structure by
            # default
            gs[m] = GS3d(dim=uigrid.dim,
                         m=m,
                         uidata=uidata,
                         vox_size=vox_size,
                         S_total=uisim.S,
                         uigrid=uigrid,
                         uimesh=uimesh)
            gs[m].s = deepcopy(S)
            print(f"GS temporal slice {m} stored\n\n"
                  "!! MONTE-CARLO ALG.300 run ended !!\n"
                  "...............................")
            break
        else:
            cbp = uisim.consider_boltzmann_probability
            sbp = uisim.s_boltz_prob
            S = mcloop_alg301(cbp, sbp, S, xinda, yinda, zinda)
        cond_1 = m % uiint.mcint_save_at_mcstep_interval == 0.0
        save_msg = False
        if m==0 or cond_1 or fully_annealed:
            gs[m] = GS3d(m=m,
                         dim=uigrid.dim,
                         uidata=uidata,
                         vox_size=vox_size,
                         S_total=uisim.S,
                         uigrid=uigrid,
                         uimesh=uimesh)
            gs[m].s = deepcopy(S)
            save_msg = True
            print(f"GS temporal slice {m} stored")
        if m % uiint.mcint_promt_display == 0:
            if not save_msg:
                print(f"Monte-Carlo temporal step = {m}")
    print('|' + 15*'-'+' MC SIM RUN COMPLETED on: ALG310' + 15*'-' + '|')
    fully_annealed = {'fully_annealed': fully_annealed,
                      'm': fully_annealed_at_m}
    print(f'Number of gs tslices: {len(gs.values())}')
    return gs, fully_annealed


'''
def mc_iterations_3d_alg310(S=None,
                            vox_size=None, xinda=None, yinda=None, zinda=None,
                            uidata=None, uigrid=None, uisim=None, uiint=None,
                            uimesh=None, verbose=False):
    S_sz0, S_sz1, S_sz2 = S.shape[0], S.shape[1], S.shape[2]
    S_sz0_list = list(range(S_sz0))
    S_sz1_list = list(range(S_sz1))
    S_sz2_list = list(range(S_sz2))
    # --------------------------------------
    fully_annealed = False
    fully_annealed_at_m = None
    gs = {}
    for m in range(uisim.mcsteps):
        if S.min() == S.max():
            print(30*'.')
            print(f'Single crystal achieved at iteration {m}.')
            fully_annealed, fully_annealed_at_m = True, m
            # Store the last temporal slice as a UPXO grain structure by
            # default
            gs[m] = GS3d(dim=uigrid.dim,
                         m=m,
                         uidata=uidata,
                         vox_size=vox_size,
                         S_total=uisim.S,
                         uigrid=uigrid,
                         uimesh=uimesh)
            gs[m].s = deepcopy(S)
            print(f"GS temporal slice {m} stored\n\n"
                  "!! MONTE-CARLO ALG.300 run ended !!\n"
                  "...............................")
            break
        else:
            for P in S_sz2_list:  # along axis 2, along plane
                for R in S_sz0_list:  # along axis 1, along row
                    for C in S_sz1_list:  # along axis 0, along column.
                        P0, R0, C0 = P, R, C
                        P1, R1, C1 = P+1, R+1, C+1
                        P2, R2, C2 = P+2, R+2, C+2
                        ssub_000 = S[zinda[P0, R0, C0],
                                     yinda[P0, R0, C0],
                                     xinda[P0, R0, C0]]
                        ssub_001 = S[zinda[P0, R0, C1],
                                     yinda[P0, R0, C1],
                                     xinda[P0, R0, C1]]
                        ssub_002 = S[zinda[P0, R0, C2],
                                     yinda[P0, R0, C2],
                                     xinda[P0, R0, C2]]
                        ssub_010 = S[zinda[P0, R1, C0],
                                     yinda[P0, R1, C0],
                                     xinda[P0, R1, C0]]
                        ssub_011 = S[zinda[P0, R1, C1],
                                     yinda[P0, R1, C1],
                                     xinda[P0, R1, C1]]
                        ssub_012 = S[zinda[P0, R1, C2],
                                     yinda[P0, R1, C2],
                                     xinda[P0, R1, C2]]
                        ssub_020 = S[zinda[P0, R2, C0],
                                     yinda[P0, R2, C0],
                                     xinda[P0, R2, C0]]
                        ssub_021 = S[zinda[P0, R2, C1],
                                     yinda[P0, R2, C1],
                                     xinda[P0, R2, C1]]
                        ssub_022 = S[zinda[P0, R2, C2],
                                     yinda[P0, R2, C2],
                                     xinda[P0, R2, C2]]

                        ssub_100 = S[zinda[P1, R0, C0],
                                     yinda[P1, R0, C0],
                                     xinda[P1, R0, C0]]
                        ssub_101 = S[zinda[P1, R0, C1],
                                     yinda[P1, R0, C1],
                                     xinda[P1, R0, C1]]
                        ssub_102 = S[zinda[P1, R0, C2],
                                     yinda[P1, R0, C2],
                                     xinda[P1, R0, C2]]
                        ssub_110 = S[zinda[P1, R1, C0],
                                     yinda[P1, R1, C0],
                                     xinda[P1, R1, C0]]
                        ssub_111_a = S[zinda[P1, R1, C1],
                                       yinda[P1, R1, C1],
                                       xinda[P1, R1, C1]]
                        ssub_112 = S[zinda[P1, R1, C2],
                                     yinda[P1, R1, C2],
                                     xinda[P1, R1, C2]]
                        ssub_120 = S[zinda[P1, R2, C0],
                                     yinda[P1, R2, C0],
                                     xinda[P1, R2, C0]]
                        ssub_121 = S[zinda[P1, R2, C1],
                                     yinda[P1, R2, C1],
                                     xinda[P1, R2, C1]]
                        ssub_122 = S[zinda[P1, R2, C2],
                                     yinda[P1, R2, C2],
                                     xinda[P1, R2, C2]]

                        ssub_200 = S[zinda[P2, R0, C0],
                                     yinda[P2, R0, C0],
                                     xinda[P2, R0, C0]]
                        ssub_201 = S[zinda[P2, R0, C1],
                                     yinda[P2, R0, C1],
                                     xinda[P2, R0, C1]]
                        ssub_202 = S[zinda[P2, R0, C2],
                                     yinda[P2, R0, C2],
                                     xinda[P2, R0, C2]]
                        ssub_210 = S[zinda[P2, R1, C0],
                                     yinda[P2, R1, C0],
                                     xinda[P2, R1, C0]]
                        ssub_211 = S[zinda[P2, R1, C1],
                                     yinda[P2, R1, C1],
                                     xinda[P2, R1, C1]]
                        ssub_212 = S[zinda[P2, R1, C2],
                                     yinda[P2, R1, C2],
                                     xinda[P2, R1, C2]]
                        ssub_220 = S[zinda[P2, R2, C0],
                                     yinda[P2, R2, C0],
                                     xinda[P2, R2, C0]]
                        ssub_221 = S[zinda[P2, R2, C1],
                                     yinda[P2, R2, C1],
                                     xinda[P2, R2, C1]]
                        ssub_222 = S[zinda[P2, R2, C2],
                                     yinda[P2, R2, C2],
                                     xinda[P2, R2, C2]]

                        Neigh = [ssub_000, ssub_001, ssub_002,
                                 ssub_010, ssub_011, ssub_012,
                                 ssub_020, ssub_021, ssub_022,
                                 ssub_100, ssub_101, ssub_102,
                                 ssub_110, ssub_111_a, ssub_112,
                                 ssub_120, ssub_121, ssub_122,
                                 ssub_200, ssub_201, ssub_202,
                                 ssub_210, ssub_211, ssub_212,
                                 ssub_220, ssub_221, ssub_222]
                        if min(Neigh) != max(Neigh):
                            DelH1 = float(ssub_111_a == ssub_000) + \
                                float(ssub_111_a == ssub_001) + \
                                float(ssub_111_a == ssub_002) + \
                                float(ssub_111_a == ssub_010) + \
                                float(ssub_111_a == ssub_011) + \
                                float(ssub_111_a == ssub_012) + \
                                float(ssub_111_a == ssub_020) + \
                                float(ssub_111_a == ssub_021) + \
                                float(ssub_111_a == ssub_022) + \
                                float(ssub_111_a == ssub_100) + \
                                float(ssub_111_a == ssub_101) + \
                                float(ssub_111_a == ssub_102) + \
                                float(ssub_111_a == ssub_110) + \
                                float(ssub_111_a == ssub_112) + \
                                float(ssub_111_a == ssub_120) + \
                                float(ssub_111_a == ssub_121) + \
                                float(ssub_111_a == ssub_122) + \
                                float(ssub_111_a == ssub_200) + \
                                float(ssub_111_a == ssub_201) + \
                                float(ssub_111_a == ssub_202) + \
                                float(ssub_111_a == ssub_210) + \
                                float(ssub_111_a == ssub_211) + \
                                float(ssub_111_a == ssub_212) + \
                                float(ssub_111_a == ssub_220) + \
                                float(ssub_111_a == ssub_221) + \
                                float(ssub_111_a == ssub_222)
                            # ---------------------------------------------
                            # If the sampling is to be selected without weightage to dominant neighbour state, then:
                            Neigh = list(set([x for x in Neigh if x != ssub_111_a]))
                            # If the sampling is to be selected with weightage to dominant neighbour state, then:
                            # Neigh = [x for x in Neigh if x != ssub_111_a]
                            # ---------------------------------------------
                            ssub_111_b = sample_rand(Neigh, 1)[0]
                            DelH2 = float(ssub_111_b == ssub_000) + \
                                float(ssub_111_b == ssub_001) + \
                                float(ssub_111_b == ssub_002) + \
                                float(ssub_111_b == ssub_010) + \
                                float(ssub_111_b == ssub_011) + \
                                float(ssub_111_b == ssub_012) + \
                                float(ssub_111_b == ssub_020) + \
                                float(ssub_111_b == ssub_021) + \
                                float(ssub_111_b == ssub_022) + \
                                float(ssub_111_b == ssub_100) + \
                                float(ssub_111_b == ssub_101) + \
                                float(ssub_111_b == ssub_102) + \
                                float(ssub_111_b == ssub_110) + \
                                float(ssub_111_b == ssub_112) + \
                                float(ssub_111_b == ssub_120) + \
                                float(ssub_111_b == ssub_121) + \
                                float(ssub_111_b == ssub_122) + \
                                float(ssub_111_b == ssub_200) + \
                                float(ssub_111_b == ssub_201) + \
                                float(ssub_111_b == ssub_202) + \
                                float(ssub_111_b == ssub_210) + \
                                float(ssub_111_b == ssub_211) + \
                                float(ssub_111_b == ssub_212) + \
                                float(ssub_111_b == ssub_220) + \
                                float(ssub_111_b == ssub_221) + \
                                float(ssub_111_b == ssub_222)
                            if DelH2 >= DelH1:
                                # S[P, R, C] = ssub_111_b
                                _p = zinda[P1, R1, C1]
                                _r = yinda[P1, R1, C1]
                                _c = xinda[P1, R1, C1]
                                S[_p, _r, _c] = ssub_111_b
                            elif uisim.consider_boltzmann_probability:
                                if uisim.s_boltz_prob[int(ssub_111_b-1)] < rand.random():
                                    S[P, R, C] = ssub_111_b
                if verbose:
                    if m % uiint.mcint_promt_display == 0:
                        print("Annealing step no.", m, "Kernel core in slice. ", P, "/", S_sz2)
        cond_1 = m % uiint.mcint_save_at_mcstep_interval == 0.0
        save_msg = False
        if m==0 or cond_1 or fully_annealed:
            gs[m] = GS3d(m=m,
                         dim=uigrid.dim,
                         uidata=uidata,
                         vox_size=vox_size,
                         S_total=uisim.S,
                         uigrid=uigrid,
                         uimesh=uimesh)
            gs[m].s = deepcopy(S)
            save_msg = True
            print(f"GS temporal slice {m} stored")
        if m % uiint.mcint_promt_display == 0:
            if not save_msg:
                print(f"Monte-Carlo temporal step = {m}")
    print('|' + 15*'-'+' MC SIM RUN COMPLETED on: ALG310' + 15*'-' + '|')
    fully_annealed = {'fully_annealed': fully_annealed,
                      'm': fully_annealed_at_m}
    print(f'Number of gs tslices: {len(gs.values())}')
    return gs, fully_annealed

'''
