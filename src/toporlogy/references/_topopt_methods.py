"""Bibliography: methods added on 21 Nov 2026 (stress, buckling, GCMMA, SiMPL, PDE filter, aggregation, benchmarking).

Status of every entry (docs/literatur_2026-11-21.md): the network policy of the session that added them blocked
publisher and arXiv pages, so NO full text was read. "record" in a note means that title, authors, venue and DOI /
arXiv number were confirmed from search results of the publisher, arXiv or an institutional repository; "memory" means
the entry was written from memory and not confirmed. Formulas in the code that cite these entries say where they
come from (first principles, public reference code, or memory).
"""
from toporlogy.references import Reference

REFERENCES: tuple[Reference, ...] = (
    # --- aggregation ------------------------------------------------------------------------
    Reference("kreisselmeier_steinhauser_1979", "G. Kreisselmeier, R. Steinhauser", 1979,
              "Systematic control design by optimizing a vector performance index",
              "IFAC Proceedings Volumes 12(7):113-117", note="KS function; memory, not confirmed"),
    Reference("kennedy_hicken_2015", "G. J. Kennedy, J. E. Hicken", 2015, "Improved constraint-aggregation methods",
              "Computer Methods in Applied Mechanics and Engineering 289:332-354", doi="10.1016/j.cma.2015.02.017",
              note="induced aggregates; record (venue, pages, DOI), title from memory"),
    Reference("le_2010", "C. Le, J. Norato, T. Bruns, C. Ha, D. Tortorelli", 2010,
              "Stress-based topology optimization for continua",
              "Structural and Multidisciplinary Optimization 41(4):605-620", doi="10.1007/s00158-009-0440-y",
              note="qp relaxed stress, regional P-norm, adaptive constraint scaling; record"),
    # --- stress -----------------------------------------------------------------------------
    Reference("bruggi_2008", "M. Bruggi", 2008,
              "On an alternative approach to stress constraints relaxation in topology optimization",
              "Structural and Multidisciplinary Optimization 36(2):125-141",
              note="qp relaxation; memory, not confirmed"),
    Reference("senhora_2020", "F. V. Senhora, O. Giraldo-Londoño, I. F. M. Menezes, G. H. Paulino", 2020,
              "Topology optimization with local stress constraints: a stress aggregation-free approach",
              "Structural and Multidisciplinary Optimization 62(4):1639-1668", doi="10.1007/s00158-020-02573-9",
              note="augmented Lagrangian for local stress constraints; record"),
    Reference("giraldo_londono_paulino_2021", "O. Giraldo-Londoño, G. H. Paulino", 2021,
              "PolyStress: a Matlab implementation for local stress-constrained topology optimization using the "
              "augmented Lagrangian method", "Structural and Multidisciplinary Optimization 63(4)",
              doi="10.1007/s00158-020-02760-8",
              note="polynomial vanishing constraint m_E Λ (Λ² + 1); record, form seen through a citing paper"),
    Reference("da_silva_2021", "G. A. da Silva, N. Aage, A. T. Beck, O. Sigmund", 2021,
              "Local versus global stress constraint strategies in topology optimization: a comparative study",
              "Int. J. Numerical Methods in Engineering 122(21):6003-6036", doi="10.1002/nme.6781",
              note="comparison of aggregated and augmented-Lagrangian strategies; record"),
    Reference("bertsekas_1999", "D. P. Bertsekas", 1999, "Nonlinear Programming", "Athena Scientific", edition="2nd",
              note="augmented Lagrangian for inequality constraints; memory"),
    # --- buckling ---------------------------------------------------------------------------
    Reference("ferrari_sigmund_2019", "F. Ferrari, O. Sigmund", 2019,
              "Revisiting topology optimization with buckling constraints",
              "Structural and Multidisciplinary Optimization 59(5):1401-1415", doi="10.1007/s00158-019-02253-3",
              note="consistent buckling sensitivities (adjoint term), aggregation; record"),
    Reference("ferrari_sigmund_guest_2021", "F. Ferrari, O. Sigmund, J. K. Guest", 2021,
              "Topology optimization with linearized buckling criteria in 250 lines of Matlab",
              "Structural and Multidisciplinary Optimization 63(6):3045-3066", doi="10.1007/s00158-021-02854-x",
              note="topBuck250: separate stress-stiffness interpolation, KS of μ = 1/λ; record"),
    # --- optimizers -------------------------------------------------------------------------
    Reference("svanberg_2002", "K. Svanberg", 2002,
              "A class of globally convergent optimization methods based on conservative convex separable "
              "approximations", "SIAM Journal on Optimization 12(2):555-573",
              note="GCMMA / CCSA; memory, not confirmed"),
    Reference("svanberg_2007", "K. Svanberg", 2007, "MMA and GCMMA - two methods for nonlinear optimization",
              "KTH Royal Institute of Technology, technical note", url="https://people.kth.se/~krille/mmagcmma.pdf",
              note="GCMMA approximations and inner iterations; record of the URL, text not read"),
    Reference("kim_2025", "D. Kim, B. S. Lazarov, T. M. Surowiec, B. Keith", 2025,
              "A simple introduction to the SiMPL method for density-based topology optimization",
              "Structural and Multidisciplinary Optimization; arXiv:2411.19421", url="https://arxiv.org/abs/2411.19421",
              note="sigmoidal mirror descent with a projected latent variable; record of the arXiv entry"),
    Reference("keith_2025", "B. Keith, D. Kim, B. S. Lazarov, T. M. Surowiec", 2025,
              "Analysis of the SiMPL method for density-based topology optimization",
              "SIAM Journal on Optimization; arXiv:2409.19341", url="https://arxiv.org/abs/2409.19341",
              note="convergence analysis, Armijo and Bregman backtracking; record of the arXiv entry"),
    # --- filters ----------------------------------------------------------------------------
    Reference("lazarov_sigmund_2011", "B. S. Lazarov, O. Sigmund", 2011,
              "Filters in topology optimization based on Helmholtz-type differential equations",
              "Int. J. Numerical Methods in Engineering 86(6):765-781", doi="10.1002/nme.3072",
              note="PDE filter −r²Δx̃ + x̃ = x; record (DOI from a reference list); r = R/(2√3) from memory"),
    # --- practice ---------------------------------------------------------------------------
    Reference("sigmund_2022", "O. Sigmund", 2022, "On benchmarking and good scientific practise in topology "
              "optimization", "Structural and Multidisciplinary Optimization 65:315",
              doi="10.1007/s00158-022-03427-2", note="compare thresholded (0/1) designs at equal resolution; record"),
)
