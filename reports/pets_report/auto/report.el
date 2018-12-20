(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("IEEEtran" "10pt" "journal")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("subfig" "caption=false" "font=footnotesize")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "IEEEtran"
    "IEEEtran10"
    "amsmath"
    "amsfonts"
    "amssymb"
    "hyperref"
    "graphicx"
    "subfig")
   (TeX-add-symbols
    "endthebibliography")
   (LaTeX-add-labels
    "sec:introduction"
    "sec:method"
    "fig:pets"
    "fig:unet"
    "sec:model"
    "sec:results"
    "tab:dataset-summary"
    "sec:discussion"
    "sec:impl-instr")
   (LaTeX-add-bibliographies
    "vision_project"))
 :latex)

