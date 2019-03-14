(TeX-add-style-hook
 "figures"
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
    "subfig"
    "siunitx")
   (TeX-add-symbols
    '("norm" 1))
   (LaTeX-add-labels
    "fig:object-positions-in-label-space"
    "fig:known-label-space-boundaries"
    "fig:augmentation-without-resampling"
    "fig:resampling-from-known-label-space-boundary"
    "fig:label-space"
    "fig:training"
    "eq:1"))
 :latex)

