(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("IEEEtran" "10pt" "journal")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("subfig" "caption=false" "font=footnotesize")))
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
   (LaTeX-add-labels
    "sec:introduction"
    "fig:traditional-graph"
    "fig:artifice-graph"
    "fig:dependency-graphs"
    "sec:governing-principles"
    "sec:related-work"
    "fig:gyros"
    "fig:coupled-spheres"
    "fig:example-experiments"
    "sec:method"
    "sec:simulated-experiments"
    "sec:conclusion"
    "sec:acknowledgment")
   (LaTeX-add-bibliographies
    "IEEEabrv"
    " report"))
 :latex)

