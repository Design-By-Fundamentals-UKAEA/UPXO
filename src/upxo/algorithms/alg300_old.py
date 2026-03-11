def mc_iterations_3d_alg300(self):
    """


    Returns
    -------
    None.

    """

    # _a, _b, _c, _d, _e = self.build_NLM()
    # NLM_00, NLM_01, NLM_02, NLM_03, NLM_04 = _a
    # NLM_10, NLM_11, NLM_12, NLM_13, NLM_14 = _b
    # NLM_20, NLM_21, NLM_22, NLM_23, NLM_24 = _c
    # NLM_30, NLM_31, NLM_32, NLM_33, NLM_34 = _d
    # NLM_40, NLM_41, NLM_42, NLM_43, NLM_44 = _e

    S_sz0, S_sz1, S_sz2 = self.S.shape[0], self.S.shape[1], self.S.shape[2]
    S_sz0_list, S_sz1_list = list(range(S_sz0)), list(range(S_sz1))
    S_sz2_list = list(range(S_sz2))
    # --------------------------------------
    if self.uisim.NL == 1:
        NLM_000 = self.NLM_nd[0, 0, 0]
        NLM_001 = self.NLM_nd[0, 0, 1]
        NLM_002 = self.NLM_nd[0, 0, 2]
        NLM_010 = self.NLM_nd[0, 1, 0]
        NLM_011 = self.NLM_nd[0, 1, 1]
        NLM_012 = self.NLM_nd[0, 1, 2]
        NLM_020 = self.NLM_nd[0, 2, 0]
        NLM_021 = self.NLM_nd[0, 2, 1]
        NLM_022 = self.NLM_nd[0, 2, 2]

        NLM_100 = self.NLM_nd[1, 0, 0]
        NLM_101 = self.NLM_nd[1, 0, 1]
        NLM_102 = self.NLM_nd[1, 0, 2]
        NLM_110 = self.NLM_nd[1, 1, 0]
        # NLM_111 = self.NLM_nd[1, 1, 1]
        NLM_112 = self.NLM_nd[1, 1, 2]
        NLM_120 = self.NLM_nd[1, 2, 0]
        NLM_121 = self.NLM_nd[1, 2, 1]
        NLM_122 = self.NLM_nd[1, 2, 2]

        NLM_200 = self.NLM_nd[2, 0, 0]
        NLM_201 = self.NLM_nd[2, 0, 1]
        NLM_202 = self.NLM_nd[2, 0, 2]
        NLM_210 = self.NLM_nd[2, 1, 0]
        NLM_211 = self.NLM_nd[2, 1, 1]
        NLM_212 = self.NLM_nd[2, 1, 2]
        NLM_220 = self.NLM_nd[2, 2, 0]
        NLM_221 = self.NLM_nd[2, 2, 1]
        NLM_222 = self.NLM_nd[2, 2, 2]
    # --------------------------------------
    xinda = self.xinda
    yinda = self.yinda
    zinda = self.zinda
    # sa = self.sa
    # --------------------------------------
    fully_annealed = False
    # --------------------------------------
    # Begin modified Markov-Chain iterations
    for m in range(self.uisim.mcsteps):
        if self.S.min() == self.S.max():
            print('Domain fully annealed at iteration', m)
            break
        else:
            for P in S_sz2_list:  # along axis 2, along plane
                for R in S_sz0_list:  # along axis 1, along row
                    for C in S_sz1_list:  # along axis 0, along column
                        ssub_000 = self.S[zinda[P+0, R+0, C+0],
                                            yinda[P+0, R+0, C+0],
                                            xinda[P+0, R+0, C+0]]
                        ssub_001 = self.S[zinda[P+0, R+0, C+1],
                                            yinda[P+0, R+0, C+1],
                                            xinda[P+0, R+0, C+1]]
                        ssub_002 = self.S[zinda[P+0, R+0, C+2],
                                            yinda[P+0, R+0, C+2],
                                            xinda[P+0, R+0, C+2]]
                        ssub_010 = self.S[zinda[P+0, R+1, C+0],
                                            yinda[P+0, R+1, C+0],
                                            xinda[P+0, R+1, C+0]]
                        ssub_011 = self.S[zinda[P+0, R+1, C+1],
                                            yinda[P+0, R+1, C+1],
                                            xinda[P+0, R+1, C+1]]
                        ssub_012 = self.S[zinda[P+0, R+1, C+2],
                                            yinda[P+0, R+1, C+2],
                                            xinda[P+0, R+1, C+2]]
                        ssub_020 = self.S[zinda[P+0, R+2, C+0],
                                            yinda[P+0, R+2, C+0],
                                            xinda[P+0, R+2, C+0]]
                        ssub_021 = self.S[zinda[P+0, R+2, C+1],
                                            yinda[P+0, R+2, C+1],
                                            xinda[P+0, R+2, C+1]]
                        ssub_022 = self.S[zinda[P+0, R+2, C+2],
                                            yinda[P+0, R+2, C+2],
                                            xinda[P+0, R+2, C+2]]

                        ssub_100 = self.S[zinda[P+1, R+0, C+0],
                                            yinda[P+1, R+0, C+0],
                                            xinda[P+1, R+0, C+0]]
                        ssub_101 = self.S[zinda[P+1, R+0, C+1],
                                            yinda[P+1, R+0, C+1],
                                            xinda[P+1, R+0, C+1]]
                        ssub_102 = self.S[zinda[P+1, R+0, C+2],
                                            yinda[P+1, R+0, C+2],
                                            xinda[P+1, R+0, C+2]]
                        ssub_110 = self.S[zinda[P+1, R+1, C+0],
                                            yinda[P+1, R+1, C+0],
                                            xinda[P+1, R+1, C+0]]
                        ssub_111_a = self.S[zinda[P+1, R+1, C+1],
                                            yinda[P+1, R+1, C+1],
                                            xinda[P+1, R+1, C+1]]
                        ssub_112 = self.S[zinda[P+1, R+1, C+2],
                                            yinda[P+1, R+1, C+2],
                                            xinda[P+1, R+1, C+2]]
                        ssub_120 = self.S[zinda[P+1, R+2, C+0],
                                            yinda[P+1, R+2, C+0],
                                            xinda[P+1, R+2, C+0]]
                        ssub_121 = self.S[zinda[P+1, R+2, C+1],
                                            yinda[P+1, R+2, C+1],
                                            xinda[P+1, R+2, C+1]]
                        ssub_122 = self.S[zinda[P+1, R+2, C+2],
                                            yinda[P+1, R+2, C+2],
                                            xinda[P+1, R+2, C+2]]

                        ssub_200 = self.S[zinda[P+2, R+0, C+0],
                                            yinda[P+2, R+0, C+0],
                                            xinda[P+2, R+0, C+0]]
                        ssub_201 = self.S[zinda[P+2, R+0, C+1],
                                            yinda[P+2, R+0, C+1],
                                            xinda[P+2, R+0, C+1]]
                        ssub_202 = self.S[zinda[P+2, R+0, C+2],
                                            yinda[P+2, R+0, C+2],
                                            xinda[P+2, R+0, C+2]]
                        ssub_210 = self.S[zinda[P+2, R+1, C+0],
                                            yinda[P+2, R+1, C+0],
                                            xinda[P+2, R+1, C+0]]
                        ssub_211 = self.S[zinda[P+2, R+1, C+1],
                                            yinda[P+2, R+1, C+1],
                                            xinda[P+2, R+1, C+1]]
                        ssub_212 = self.S[zinda[P+2, R+1, C+2],
                                            yinda[P+2, R+1, C+2],
                                            xinda[P+2, R+1, C+2]]
                        ssub_220 = self.S[zinda[P+2, R+2, C+0],
                                            yinda[P+2, R+2, C+0],
                                            xinda[P+2, R+2, C+0]]
                        ssub_221 = self.S[zinda[P+2, R+2, C+1],
                                            yinda[P+2, R+2, C+1],
                                            xinda[P+2, R+2, C+1]]
                        ssub_222 = self.S[zinda[P+2, R+2, C+2],
                                            yinda[P+2, R+2, C+2],
                                            xinda[P+2, R+2, C+2]]

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
                            DelH1 = NLM_000 * float(ssub_111_a == ssub_000) + \
                                NLM_001 * float(ssub_111_a == ssub_001) + \
                                NLM_002 * float(ssub_111_a == ssub_002) + \
                                NLM_010 * float(ssub_111_a == ssub_010) + \
                                NLM_011 * float(ssub_111_a == ssub_011) + \
                                NLM_012 * float(ssub_111_a == ssub_012) + \
                                NLM_020 * float(ssub_111_a == ssub_020) + \
                                NLM_021 * float(ssub_111_a == ssub_021) + \
                                NLM_022 * float(ssub_111_a == ssub_022) + \
                                NLM_100 * float(ssub_111_a == ssub_100) + \
                                NLM_101 * float(ssub_111_a == ssub_101) + \
                                NLM_102 * float(ssub_111_a == ssub_102) + \
                                NLM_110 * float(ssub_111_a == ssub_110) + \
                                NLM_112 * float(ssub_111_a == ssub_112) + \
                                NLM_120 * float(ssub_111_a == ssub_120) + \
                                NLM_121 * float(ssub_111_a == ssub_121) + \
                                NLM_122 * float(ssub_111_a == ssub_122) + \
                                NLM_200 * float(ssub_111_a == ssub_200) + \
                                NLM_201 * float(ssub_111_a == ssub_201) + \
                                NLM_202 * float(ssub_111_a == ssub_202) + \
                                NLM_210 * float(ssub_111_a == ssub_210) + \
                                NLM_211 * float(ssub_111_a == ssub_211) + \
                                NLM_212 * float(ssub_111_a == ssub_212) + \
                                NLM_220 * float(ssub_111_a == ssub_220) + \
                                NLM_221 * float(ssub_111_a == ssub_221) + \
                                NLM_222 * float(ssub_111_a == ssub_222)
                            # ---------------------------------------------
                            # If the sampling is to be selected without weightage to dominant neighbour state, then:
                            Neigh = list(set([x for x in Neigh if x != ssub_111_a]))
                            # If the sampling is to be selected with weightage to dominant neighbour state, then:
                            # Neigh = [x for x in Neigh if x != ssub_111_a]
                            # ---------------------------------------------
                            ssub_111_b = sample_rand(Neigh, 1)[0]
                            DelH2 = NLM_000 * float(ssub_111_b == ssub_000) + \
                                NLM_001 * float(ssub_111_b == ssub_001) + \
                                NLM_002 * float(ssub_111_b == ssub_002) + \
                                NLM_010 * float(ssub_111_b == ssub_010) + \
                                NLM_011 * float(ssub_111_b == ssub_011) + \
                                NLM_012 * float(ssub_111_b == ssub_012) + \
                                NLM_020 * float(ssub_111_b == ssub_020) + \
                                NLM_021 * float(ssub_111_b == ssub_021) + \
                                NLM_022 * float(ssub_111_b == ssub_022) + \
                                NLM_100 * float(ssub_111_b == ssub_100) + \
                                NLM_101 * float(ssub_111_b == ssub_101) + \
                                NLM_102 * float(ssub_111_b == ssub_102) + \
                                NLM_110 * float(ssub_111_b == ssub_110) + \
                                NLM_112 * float(ssub_111_b == ssub_112) + \
                                NLM_120 * float(ssub_111_b == ssub_120) + \
                                NLM_121 * float(ssub_111_b == ssub_121) + \
                                NLM_122 * float(ssub_111_b == ssub_122) + \
                                NLM_200 * float(ssub_111_b == ssub_200) + \
                                NLM_201 * float(ssub_111_b == ssub_201) + \
                                NLM_202 * float(ssub_111_b == ssub_202) + \
                                NLM_210 * float(ssub_111_b == ssub_210) + \
                                NLM_211 * float(ssub_111_b == ssub_211) + \
                                NLM_212 * float(ssub_111_b == ssub_212) + \
                                NLM_220 * float(ssub_111_b == ssub_220) + \
                                NLM_221 * float(ssub_111_b == ssub_221) + \
                                NLM_222 * float(ssub_111_b == ssub_222)
                            if DelH2 >= DelH1:
                                # S[P, R, C] = ssub_111_b
                                _p = zinda[P+1, R+1, C+1]
                                _r = yinda[P+1, R+1, C+1]
                                _c = xinda[P+1, R+1, C+1]
                                self.S[_p, _r, _c] = ssub_111_b
                            elif self.uisim.consider_boltzmann_probability:
                                if self.uisim.s_boltz_prob[int(ssub_111_b-1)] < rand.random():
                                    self.S[P, R, C] = ssub_111_b
        if self.display_messages:
            if m % self.uiint.mcint_promt_display == 0:
                print("Annealing step no.", m)
        cond_1 = m % self.uiint.mcint_save_at_mcstep_interval == 0.0
        if cond_1 or fully_annealed:
            # self.tslices.append(m)
            # Create and add grain structure data structure template
            self.add_gs_data_structure_template(m=m,
                                                dim=self.uigrid.dim)
            self.gs[m].s = deepcopy(self.S)
            if self.display_messages:
                print(f'State updated @ mc step {m}')
            print('__________________________')
