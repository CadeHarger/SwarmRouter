from mealpy.bio_based import BBO, EOA, IWO, SBO, SMA, TPO, VCS, WHO
from mealpy.evolutionary_based import CRO, DE, EP, ES, FPA, GA, MA
from mealpy.human_based import BRO, BSO, CA, CHIO, FBIO, GSKA, ICA, LCO, QSA, SARO, SSDO, TLO
from mealpy.math_based import AOA, CEM, CGO, GBO, HC, PSS, SCA
from mealpy.music_based import HS
from mealpy.physics_based import ArchOA, ASO, EFO, EO, HGSO, MVO, NRO, SA, TWO, WDO
from mealpy.system_based import AEO, GCO, WCA
from mealpy.swarm_based import ABC, ACOR, ALO, AO, BA, BeesA, BES, BFO, BSA, COA, CSA, CSO, DO, EHO, FA, FFA, FOA, GOA, GWO, HGS
from mealpy.swarm_based import HHO, JA, MFO, MRFO, MSA, NMRA, PFA, PSO, SFO, SHO, SLO, SRSR, SSA, SSO, SSpiderA, SSpiderO, WOA


paras_bbo = {
    
    "pop_size": 50,
    "p_m": 0.01,
    "elites": 2,
}
paras_eoa = {
    
    "pop_size": 50,
    "p_c": 0.9,
    "p_m": 0.01,
    "n_best": 2,
    "alpha": 0.98,
    "beta": 0.9,
    "gamma": 0.9,
}
paras_iwo = {
    
    "pop_size": 50,
    "seed_min": 3,
    "seed_max": 9,
    "exponent": 3,
    "sigma_start": 0.6,
    "sigma_end": 0.01,
}
paras_sbo = {
    
    "pop_size": 50,
    "alpha": 0.9,
    "p_m": 0.05,
    "psw": 0.02,
}
paras_sma = {
    
    "pop_size": 50,
    "p_t": 0.03,
}
paras_vcs = {
    
    "pop_size": 50,
    "lamda": 0.5,
    "sigma": 0.3,
}
paras_who = {
    
    "pop_size": 50,
    "n_explore_step": 3,
    "n_exploit_step": 3,
    "eta": 0.15,
    "p_hi": 0.9,
    "local_alpha": 0.9,
    "local_beta": 0.3,
    "global_alpha": 0.2,
    "global_beta": 0.8,
    "delta_w": 2.0,
    "delta_c": 2.0,
}
paras_cro = {
    
    "pop_size": 50,
    "po": 0.4,
    "Fb": 0.9,
    "Fa": 0.1,
    "Fd": 0.1,
    "Pd": 0.5,
    "GCR": 0.1,
    "gamma_min": 0.02,
    "gamma_max": 0.2,
    "n_trials": 5,
}
paras_ocro = dict(paras_cro)
paras_ocro["restart_count"] = 5

paras_de = {
    
    "pop_size": 50,
    "wf": 0.7,
    "cr": 0.9,
    "strategy": 0,
}
paras_jade = {
    
    "pop_size": 50,
    "miu_f": 0.5,
    "miu_cr": 0.5,
    "pt": 0.1,
    "ap": 0.1,
}
paras_sade = {
    
    "pop_size": 50,
}
paras_shade = paras_lshade = {
    
    "pop_size": 50,
    "miu_f": 0.5,
    "miu_cr": 0.5,
}
paras_sap_de = {
    
    "pop_size": 50,
    "branch": "ABS"
}
paras_ep = paras_levy_ep = {
    
    "pop_size": 50,
    "bout_size": 0.05
}
paras_es = paras_levy_es = {
    
    "pop_size": 50,
    "lamda": 0.75
}
paras_fpa = {
    
    "pop_size": 50,
    "p_s": 0.8,
    "levy_multiplier": 0.2
}
paras_ga = {
    
    "pop_size": 50,
    "pc": 0.9,
    "pm": 0.05,
}
paras_single_ga = {
    
    "pop_size": 50,
    "pc": 0.9,
    "pm": 0.8,
    "selection": "roulette",
    "crossover": "uniform",
    "mutation": "swap",
}
paras_multi_ga = {
    
    "pop_size": 50,
    "pc": 0.9,
    "pm": 0.05,
    "selection": "roulette",
    "crossover": "uniform",
    "mutation": "swap",
}
paras_ma = {
    
    "pop_size": 50,
    "pc": 0.85,
    "pm": 0.15,
    "p_local": 0.5,
    "max_local_gens": 10,
    "bits_per_param": 4,
}

paras_bro = {
    
    "pop_size": 50,
    "threshold": 3,
}
paras_improved_bso = {
    
    "pop_size": 50,
    "m_clusters": 5,
    "p1": 0.2,
    "p2": 0.8,
    "p3": 0.4,
    "p4": 0.5,
}
paras_bso = dict(paras_improved_bso)
paras_bso["slope"] = 20
paras_ca = {
    
    "pop_size": 50,
    "accepted_rate": 0.15,
}
paras_chio = {
    
    "pop_size": 50,
    "brr": 0.15,
    "max_age": 3
}
paras_fbio = {
    
    "pop_size": 50,
}
paras_base_gska = {
    
    "pop_size": 50,
    "pb": 0.1,
    "kr": 0.9,
}
paras_gska = {
    
    "pop_size": 50,
    "pb": 0.1,
    "kf": 0.5,
    "kr": 0.9,
    "kg": 5,
}
paras_ica = {
    
    "pop_size": 50,
    "empire_count": 5,
    "assimilation_coeff": 1.5,
    "revolution_prob": 0.05,
    "revolution_rate": 0.1,
    "revolution_step_size": 0.1,
    "zeta": 0.1,
}
paras_lco = {
    
    "pop_size": 50,
    "r1": 2.35,
}
paras_improved_lco = {
    
    "pop_size": 50,
}
paras_qsa = {
    
    "pop_size": 50,
}
paras_saro = {
    
    "pop_size": 50,
    "se": 0.5,
    "mu": 15
}
paras_ssdo = {
    
    "pop_size": 50,
}
paras_tlo = {
    
    "pop_size": 50,
}
paras_improved_tlo = {
    
    "pop_size": 50,
    "n_teachers": 5,
}

paras_aoa = {
    
    "pop_size": 50,
    "alpha": 5,
    "miu": 0.5,
    "moa_min": 0.2,
    "moa_max": 0.9,
}
paras_cem = {
    
    "pop_size": 50,
    "n_best": 20,
    "alpha": 0.7,
}
paras_cgo = {
    
    "pop_size": 50,
}
paras_gbo = {
    
    "pop_size": 50,
    "pr": 0.5,
    "beta_min": 0.2,
    "beta_max": 1.2,
}
paras_hc = {
    
    "pop_size": 50,
    "neighbour_size": 50
}
paras_swarm_hc = {
    
    "pop_size": 50,
    "neighbour_size": 10
}
paras_pss = {
    
    "pop_size": 50,
    "acceptance_rate": 0.8,
    "sampling_method": "LHS",
}
paras_sca = {
    
    "pop_size": 50,
}

paras_hs = {
    
    "pop_size": 50,
    "c_r": 0.95,
    "pa_r": 0.05
}

paras_aeo = {
    
    "pop_size": 50,
}
paras_gco = {
    
    "pop_size": 50,
    "cr": 0.7,
    "wf": 1.25,
}
paras_wca = {
    
    "pop_size": 50,
    "nsr": 4,
    "wc": 2.0,
    "dmax": 1e-6
}

paras_archoa = {
    
    "pop_size": 50,
    "c1": 2,
    "c2": 5,
    "c3": 2,
    "c4": 0.5,
    "acc_max": 0.9,
    "acc_min": 0.1,
}
paras_aso = {
    
    "pop_size": 50,
    "alpha": 50,
    "beta": 0.2,
}
paras_efo = {
    
    "pop_size": 50,
    "r_rate": 0.3,
    "ps_rate": 0.85,
    "p_field": 0.1,
    "n_field": 0.45,
}
paras_eo = {
    
    "pop_size": 50,
}
paras_hgso = {
    
    "pop_size": 50,
    "n_clusters": 3,
}
paras_mvo = {
    
    "pop_size": 50,
    "wep_min": 0.2,
    "wep_max": 1.0,
}
paras_nro = {
    
    "pop_size": 50,
}
paras_sa = {
    
    "pop_size": 50,
    "max_sub_iter": 5,
    "t0": 1000,
    "t1": 1,
    "move_count": 5,
    "mutation_rate": 0.1,
    "mutation_step_size": 0.1,
    "mutation_step_size_damp": 0.99,
}
paras_two = {
    
    "pop_size": 50,
}
paras_wdo = {
    
    "pop_size": 50,
    "RT": 3,
    "g_c": 0.2,
    "alp": 0.4,
    "c_e": 0.4,
    "max_v": 0.3,
}

paras_abc = {
    
    "pop_size": 50,
    "n_elites": 16,
    "n_others": 4,
    "patch_size": 5.0,
    "patch_reduction": 0.985,
    "n_sites": 3,
    "n_elite_sites": 1,
}
paras_acor = {
    
    "pop_size": 50,
    "sample_count": 25,
    "intent_factor": 0.5,
    "zeta": 1.0,
}
paras_alo = {
    
    "pop_size": 50,
}
paras_ao = {
    
    "pop_size": 50,
}
paras_ba = {
    
    "pop_size": 50,
    "loudness": 0.8,
    "pulse_rate": 0.95,
    "pf_min": 0.,
    "pf_max": 10.,
}
paras_adaptive_ba = {
    
    "pop_size": 50,
    "loudness_min": 1.0,
    "loudness_max": 2.0,
    "pr_min": 0.15,
    "pr_max": 0.85,
    "pf_min": 0.,
    "pf_max": 10.,
}
paras_modified_ba = {
    
    "pop_size": 50,
    "pulse_rate": 0.95,
    "pf_min": 0.,
    "pf_max": 10.,
}
paras_beesa = {
    
    "pop_size": 50,
    "selected_site_ratio": 0.5,
    "elite_site_ratio": 0.4,
    "selected_site_bee_ratio": 0.1,
    "elite_site_bee_ratio": 2.0,
    "dance_radius": 0.1,
    "dance_reduction": 0.99,
}
paras_prob_beesa = {
    
    "pop_size": 50,
    "recruited_bee_ratio": 0.1,
    "dance_radius": 0.1,
    "dance_reduction": 0.99,
}
paras_bes = {
    
    "pop_size": 50,
    "a_factor": 10,
    "R_factor": 1.5,
    "alpha": 2.0,
    "c1": 2.0,
    "c2": 2.0,
}
paras_bfo = {
    
    "pop_size": 50,
    "Ci": 0.01,
    "Ped": 0.25,
    "Nc": 5,
    "Ns": 4,
    "d_attract": 0.1,
    "w_attract": 0.2,
    "h_repels": 0.1,
    "w_repels": 10,
}
paras_abfo = {
    
    "pop_size": 50,
    "C_s": 0.1,
    "C_e": 0.001,
    "Ped": 0.01,
    "Ns": 4,
    "N_adapt": 4,
    "N_split": 40,
}
paras_bsa = {
    
    "pop_size": 50,
    "ff": 10,
    "pff": 0.8,
    "c1": 1.5,
    "c2": 1.5,
    "a1": 1.0,
    "a2": 1.0,
    "fl": 0.5,
}
paras_coa = {
    
    "pop_size": 50,
    "n_coyotes": 5,
}
paras_csa = {
    
    "pop_size": 50,
    "p_a": 0.3,
}
paras_cso = {
    
    "pop_size": 50,
    "mixture_ratio": 0.15,
    "smp": 5,
    "spc": False,
    "cdc": 0.8,
    "srd": 0.15,
    "c1": 0.4,
    "w_min": 0.4,
    "w_max": 0.9,
    "selected_strategy": 1,
}
paras_do = {
    
    "pop_size": 50,
}
paras_eho = {
    
    "pop_size": 50,
    "alpha": 0.5,
    "beta": 0.5,
    "n_clans": 5,
}
paras_fa = {
    
    "pop_size": 50,
    "max_sparks": 20,
    "p_a": 0.04,
    "p_b": 0.8,
    "max_ea": 40,
    "m_sparks": 5,
}
paras_ffa = {
    
    "pop_size": 50,
    "gamma": 0.001,
    "beta_base": 2,
    "alpha": 0.2,
    "alpha_damp": 0.99,
    "delta": 0.05,
    "exponent": 2,
}
paras_foa = {
    
    "pop_size": 50,
}
paras_goa = {
    
    "pop_size": 50,
    "c_min": 0.00004,
    "c_max": 1.0,
}
paras_gwo = {
    
    "pop_size": 50,
}
paras_hgs = {
    
    "pop_size": 50,
    "PUP": 0.08,
    "LH": 10000,
}
paras_hho = {
    
    "pop_size": 50,
}
paras_ja = {
    
    "pop_size": 50,
}
paras_mfo = {
    
    "pop_size": 50,
}
paras_mrfo = {
    
    "pop_size": 50,
    "somersault_range": 2.0,
}
paras_msa = {
    
    "pop_size": 50,
    "n_best": 5,
    "partition": 0.5,
    "max_step_size": 1.0,
}
paras_nmra = {
    
    "pop_size": 50,
    "pb": 0.75,
}
paras_improved_nmra = {
    
    "pop_size": 50,
    "pb": 0.75,
    "pm": 0.01,
}
paras_pfa = {
    
    "pop_size": 50,
}
paras_pso = {
    
    "pop_size": 50,
    "c1": 2.05,
    "c2": 2.05,
    "w_min": 0.4,
    "w_max": 0.9,
}
paras_ppso = {
    
    "pop_size": 50,
}
paras_hpso_tvac = {
    
    "pop_size": 50,
    "ci": 0.5,
    "cf": 0.0,
}
paras_cpso = {
    
    "pop_size": 50,
    "c1": 2.05,
    "c2": 2.05,
    "w_min": 0.4,
    "w_max": 0.9,
}
paras_clpso = {
    
    "pop_size": 50,
    "c_local": 1.2,
    "w_min": 0.4,
    "w_max": 0.9,
    "max_flag": 7,
}
paras_sfo = {
    
    "pop_size": 50,
    "pp": 0.1,
    "AP": 4.0,
    "epsilon": 0.0001,
}
paras_improved_sfo = {
    
    "pop_size": 50,
    "pp": 0.1,
}
paras_sho = {
    
    "pop_size": 50,
    "h_factor": 5.0,
    "N_tried": 10,
}
paras_slo = paras_modified_slo = {
    
    "pop_size": 50,
}
paras_improved_slo = {
    
    "pop_size": 50,
    "c1": 1.2,
    "c2": 1.2
}
paras_srsr = {
    
    "pop_size": 50,
}
paras_ssa = {
    
    "pop_size": 50,
    "ST": 0.8,
    "PD": 0.2,
    "SD": 0.1,
}
paras_sso = {
    
    "pop_size": 50,
}
paras_sspidera = {
    
    "pop_size": 50,
    "r_a": 1.0,
    "p_c": 0.7,
    "p_m": 0.1
}
paras_sspidero = {
    
    "pop_size": 50,
    "fp_min": 0.65,
    "fp_max": 0.9
}
paras_woa = {
    
    "pop_size": 50,
}
paras_hi_woa = {
    
    "pop_size": 50,
    "feedback_max": 10
}

def get_models():
    models = []
    models.append(BBO.BaseBBO(**paras_bbo))
    models.append(BBO.OriginalBBO(**paras_bbo))
    models.append(EOA.OriginalEOA(**paras_eoa))
    models.append(IWO.OriginalIWO(**paras_eoa))
    models.append(SBO.BaseSBO(**paras_sbo))
    models.append(SBO.OriginalSBO(**paras_sbo))
    models.append(SMA.BaseSMA(**paras_sma))
    models.append(SMA.OriginalSMA(**paras_sma))
    models.append(VCS.BaseVCS(**paras_vcs))
    models.append(VCS.OriginalVCS(**paras_vcs))
    models.append(WHO.OriginalWHO(**paras_vcs))

    models.append(CRO.OriginalCRO(**paras_cro))
    models.append(CRO.OCRO(**paras_ocro))
    models.append(DE.BaseDE(**paras_de))
    models.append(DE.JADE(**paras_jade))
    models.append(DE.SADE(**paras_sade))
    models.append(DE.SHADE(**paras_shade))
    models.append(DE.L_SHADE(**paras_lshade))
    models.append(DE.SAP_DE(**paras_sap_de))
    models.append(EP.OriginalEP(**paras_ep))
    models.append(EP.LevyEP(**paras_levy_ep))
    models.append(ES.OriginalES(**paras_ep))
    models.append(ES.LevyES(**paras_levy_ep))
    models.append(FPA.OriginalFPA(**paras_fpa))
    models.append(GA.BaseGA(**paras_ga))
    models.append(GA.SingleGA(**paras_single_ga))
    models.append(GA.MultiGA(**paras_multi_ga))
    models.append(MA.OriginalMA(**paras_ma))

    models.append(BRO.BaseBRO(**paras_bro))
    models.append(BRO.OriginalBRO(**paras_bro))
    models.append(BSO.OriginalBSO(**paras_bso))
    models.append(BSO.ImprovedBSO(**paras_improved_bso))
    models.append(CA.OriginalCA(**paras_ca))
    models.append(CHIO.BaseCHIO(**paras_chio))
    models.append(CHIO.OriginalCHIO(**paras_chio))
    models.append(FBIO.BaseFBIO(**paras_fbio))
    models.append(FBIO.OriginalFBIO(**paras_fbio))
    models.append(GSKA.BaseGSKA(**paras_base_gska))
    models.append(GSKA.OriginalGSKA(**paras_gska))
    models.append(ICA.OriginalICA(**paras_ica))
    models.append(LCO.BaseLCO(**paras_lco))
    models.append(LCO.OriginalLCO(**paras_lco))
    models.append(LCO.ImprovedLCO(**paras_improved_lco))
    models.append(QSA.BaseQSA(**paras_qsa))
    models.append(QSA.OriginalQSA(**paras_qsa))
    models.append(QSA.OppoQSA(**paras_qsa))
    models.append(QSA.LevyQSA(**paras_qsa))
    models.append(QSA.ImprovedQSA(**paras_qsa))
    models.append(SARO.BaseSARO(**paras_saro))
    models.append(SARO.OriginalSARO(**paras_saro))
    models.append(SSDO.OriginalSSDO(**paras_ssdo))
    models.append(TLO.BaseTLO(**paras_tlo))
    models.append(TLO.OriginalTLO(**paras_tlo))
    models.append(TLO.ImprovedTLO(**paras_improved_tlo))

    models.append(AOA.OriginalAOA(**paras_aoa))
    models.append(CEM.OriginalCEM(**paras_cem))
    models.append(CGO.OriginalCGO(**paras_cgo))
    models.append(GBO.OriginalGBO(**paras_gbo))
    models.append(HC.OriginalHC(**paras_hc))
    models.append(HC.SwarmHC(**paras_swarm_hc))
    models.append(PSS.OriginalPSS(**paras_pss))
    models.append(SCA.OriginalSCA(**paras_sca))
    models.append(SCA.BaseSCA(**paras_sca))

    models.append(HS.BaseHS(**paras_hs))
    models.append(HS.OriginalHS(**paras_hs))

    models.append(AEO.OriginalAEO(**paras_aeo))
    models.append(AEO.EnhancedAEO(**paras_aeo))
    models.append(AEO.ModifiedAEO(**paras_aeo))
    models.append(AEO.ImprovedAEO(**paras_aeo))
    models.append(AEO.AugmentedAEO(**paras_aeo))
    models.append(GCO.BaseGCO(**paras_aeo))
    models.append(GCO.OriginalGCO(**paras_aeo))
    models.append(WCA.OriginalWCA(**paras_wca))

    models.append(ArchOA.OriginalArchOA(**paras_archoa))
    models.append(ASO.OriginalASO(**paras_aso))
    models.append(EFO.OriginalEFO(**paras_efo))
    models.append(EFO.BaseEFO(**paras_efo))
    models.append(EO.OriginalEO(**paras_eo))
    models.append(EO.AdaptiveEO(**paras_eo))
    models.append(EO.ModifiedEO(**paras_eo))
    models.append(HGSO.OriginalHGSO(**paras_hgso))
    models.append(MVO.OriginalMVO(**paras_mvo))
    models.append(NRO.OriginalNRO(**paras_nro))
    models.append(SA.OriginalSA(**paras_sa))
    models.append(TWO.OriginalTWO(**paras_two))
    models.append(TWO.OppoTWO(**paras_two))
    models.append(TWO.LevyTWO(**paras_two))
    models.append(TWO.EnhancedTWO(**paras_two))
    models.append(WDO.OriginalWDO(**paras_wdo))

    models.append(ABC.OriginalABC(**paras_abc))
    models.append(ACOR.OriginalACOR(**paras_acor))
    models.append(ALO.OriginalALO(**paras_alo))
    models.append(AO.OriginalAO(**paras_ao))
    models.append(ALO.BaseALO(**paras_alo))
    models.append(BA.OriginalBA(**paras_ba))
    models.append(BA.AdaptiveBA(**paras_adaptive_ba))
    models.append(BA.ModifiedBA(**paras_modified_ba))
    models.append(BeesA.OriginalBeesA(**paras_beesa))
    models.append(BeesA.ProbBeesA(**paras_prob_beesa))
    models.append(BES.OriginalBES(**paras_bes))
    models.append(BFO.OriginalBFO(**paras_bfo))
    models.append(BFO.ABFO(**paras_abfo))
    models.append(BSA.OriginalBSA(**paras_bsa))
    models.append(COA.OriginalCOA(**paras_coa))
    models.append(CSA.OriginalCSA(**paras_csa))
    models.append(CSO.OriginalCSO(**paras_cso))
    models.append(DO.OriginalDO(**paras_do))
    models.append(EHO.OriginalEHO(**paras_eho))
    models.append(FA.OriginalFA(**paras_fa))
    models.append(FFA.OriginalFFA(**paras_ffa))
    models.append(FOA.OriginalFOA(**paras_foa))
    models.append(FOA.BaseFOA(**paras_foa))
    models.append(FOA.WhaleFOA(**paras_foa))
    models.append(GOA.OriginalGOA(**paras_goa))
    models.append(GWO.OriginalGWO(**paras_gwo))
    models.append(GWO.RW_GWO(**paras_gwo))
    models.append(HGS.OriginalHGS(**paras_hgs))
    models.append(HHO.OriginalHHO(**paras_hho))
    models.append(JA.OriginalJA(**paras_ja))
    models.append(JA.BaseJA(**paras_ja))
    models.append(JA.LevyJA(**paras_ja))
    models.append(MFO.OriginalMFO(**paras_mfo))
    models.append(MFO.BaseMFO(**paras_mfo))
    models.append(MRFO.OriginalMRFO(**paras_mrfo))
    models.append(MSA.OriginalMSA(**paras_msa))
    models.append(NMRA.ImprovedNMRA(**paras_improved_nmra))
    models.append(NMRA.OriginalNMRA(**paras_nmra))
    models.append(PFA.OriginalPFA(**paras_pfa))
    models.append(PSO.OriginalPSO(**paras_pso))
    models.append(PSO.PPSO(**paras_ppso))
    models.append(PSO.HPSO_TVAC(**paras_hpso_tvac))
    models.append(PSO.C_PSO(**paras_cpso))
    models.append(PSO.CL_PSO(**paras_clpso))
    models.append(SFO.OriginalSFO(**paras_sfo))
    models.append(SFO.ImprovedSFO(**paras_improved_sfo))
    models.append(SHO.OriginalSHO(**paras_sho))
    models.append(SLO.OriginalSLO(**paras_slo))
    models.append(SLO.ModifiedSLO(**paras_modified_slo))
    models.append(SLO.ImprovedSLO(**paras_improved_slo))
    models.append(SRSR.OriginalSRSR(**paras_srsr))
    models.append(SSA.OriginalSSA(**paras_ssa))
    models.append(SSA.BaseSSA(**paras_ssa))
    models.append(SSO.OriginalSSO(**paras_sso))
    models.append(SSpiderA.OriginalSSpiderA(**paras_sspidera))
    models.append(SSpiderO.OriginalSSpiderO(**paras_sspidero))
    models.append(WOA.OriginalWOA(**paras_woa))
    models.append(WOA.HI_WOA(**paras_hi_woa))
    return models
