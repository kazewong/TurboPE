rule table:
    input:
        "src/data/GW150914_flowMC.npz",
        "src/data/GW150914_Bilby.dat",
        "src/data/GW170817_flowMC_1800.npz",
        "src/data/GW170817_Bilby_flat.dat"
    output:
        "src/tex/output/js_table.tex",
        "src/data/jsd.txt"
    script:
        "src/scripts/get_js.py"

rule pp:
    input:
        "src/data/combined_quantile_balance_LVK.npz"
    output:
        "src/tex/figures/ppplot.pdf",
        "src/data/pvalues.txt"
    script:
        "src/scripts/ppplots.py"

rule macros:
    output:
        "src/tex/output/macros.tex"
    script:
        "src/scripts/make_macros.py"
