(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("asme2ej" "twocolumn" "10pt")))
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
    "fig:real-world"
    "fig:bare-gyros"
    "fig:single-bare-gyro"
    "fig:bounding-boxes"
    "fig:bounding-boxes-shifted"
    "fig:gyro-image"
    "sec:experiment"
    "sec:first-approach"
    "sec:conv-regr"
    "sec:design"
    "sec:datasets"
    "sec:results"
    "fig:thumbnail_000_12x12_centered_images/centered_test_4"
    "thumbnail_000_12x12_centered_images/rand_bg_normal_test_3"
    "thumbnail_000_30x30_rand_bg_normal_images/rand_bg_normal_test_4"
    "thumbnail_000_30x30_rand_bg_normal_images/centered_test_4"
    "fig:thumbnails"
    "thumbnail_000_12x12_rand_bg_normal_images/rand_bg_normal_test_3"
    "thumbnail_000_12x12_rand_bg_normal_images/dot_001_rand_bg_normal_test_3"
    "fig:different-dot"
    "tab:network-performance"
    "sec:ground-truth-prec"
    "sec:artif-datas-gener"
    "fig:dependencies"
    "sec:conclusions"
    "sec:future-work")
   (LaTeX-add-bibliographies))
 :latex)

