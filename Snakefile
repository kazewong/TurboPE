rule table:
    input:
        "src/data/GW150914_flowMC.npz",
        "src/data/GW150914_Bilby.dat",
        "src/data/GW170817_flowMC_1800.npz",
        "src/data/GW170817_Bilby_flat.dat"
    output:
        "src/tex/output/js_table.tex"
    script:
        "src/scripts/get_js.py"
