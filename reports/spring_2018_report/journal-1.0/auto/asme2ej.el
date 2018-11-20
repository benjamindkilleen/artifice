(TeX-add-style-hook
 "asme2ej"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("asme2ej" "twocolumn" "10pt")))
   (TeX-run-style-hooks
    "latex2e"
    "asme2ej10"
    "epsfig")
   (LaTeX-add-labels
    "eq_ASME"
    "sect_figure"
    "figure_ASME"
    "fig_example1.ps"
    "fig_example2.ps"
    "fig_example3.ps"
    "fig_example4.ps"
    "table_ASME")
   (LaTeX-add-bibliographies
    "asme2e"))
 :latex)

