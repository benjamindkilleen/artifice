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
    "subfig"
    "siunitx")
   (TeX-add-symbols
    "endthebibliography"
    "vol"
    "argmax")
   (LaTeX-add-labels
    "sec:introduction"
    "fig:free-fall"
    "fig:traditional-graph"
    "fig:artifice-graph"
    "fig:dependency-graphs"
    "sec:method"
    "sec:evaluation"
    "fig:gyros"
    "fig:coupled-spheres"
    "fig:example-experiments"
    "sec:discussion"
    "sec:acknowledgment")
   (LaTeX-add-bibliographies))
 :latex)

