(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("asme2ej" "twocolumn" "10pt")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "asme2ej"
    "asme2ej10"
    "epsfig"
    "hyperref"
    "graphicx"
    "multirow"
    "array"
    "subcaption")
   (LaTeX-add-labels
    "sec:introduction"
    "sec:method"
    "fig:general-graph"
    "fig:artifice-graph"
    "fig:dependency-graphs")
   (LaTeX-add-bibliographies
    "artifice"))
 :latex)

