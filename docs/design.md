# Artifice Design Document

Artifice consists of *core functionality*, including:
* an **image labeler** for *semantic segmentation*, most likely
  [PixelAnnotationTool](https://github.com/abreheret/PixelAnnotationTool).
* an **image labeler** for *target features*, including `(x,y)` position,
  `theta` orientation, as well as other arbitrary numerical or categorical
  labels.
* an active learning **selection metric** for *segmentation*, similar to [Active
  Learning for semantic segmentation with expected
  change](https://ieeexplore.ieee.org/ielx5/6235193/6247647/06248050.pdf?tp=&arnumber=6248050&isnumber=6247647&tag=1)
* another **selection metric** for *target features*, such as those found in
  [Active Learning Literature
  Survey](http://burrsettles.com/pub/settles.activelearning.pdf).
* a dataset **augmentation** method (to be developed) that combats natural bias
  in experimental images.
* a first-step **segmentation model**, most likely based on
  [tf_unet](https://github.com/jakeret/tf_unet) from the original [U-Net
  paper](http://www.arxiv.org/pdf/1505.04597.pdf). [Pyramid
  Parsing](http://arxiv.org/abs/1612.01105) is another alternative, very good
  performance. [dwt](https://github.com/min2209/dwt) looks like a decent
  implementation, although it is still under construction.
* a second-step **target feature model**, very likely a small-scale CNN with
  regression outputs.
* **dataset export** capability, which writes the final target features
  obtained, and possibly network state for future use.

Additionally, artifice will include *peripheral functionality* for creating and
running test data. This includes:
* a [Vapory](https://github.com/Zulko/vapory)-based **experiment generator**,
  which also generates segmentation masks and target features for each
  image. See [log.md:Experimental
  Design](https://github.com/bendkill/artifice/log.md) for more info.
* an **artificial oracle** that emulates human labeling *and human error*
  by providing noisy segmentation masks and labels on the Vapory data.
* a **performance evaluation** tool, comparing artifice output with actual
  target data.

## Workflow
1. Create the **experiment generator** in
   [test_utils](https://github.com/bendkill/artifice/test_utils), using
   [Vapory](https://github.com/Zulko/vapory). Focus on the *find the ball*
   experiment.
2. Create a **semantic segmentation** workflow inside artifice. This could
   consist of [tf_unet](https://github.com/jakeret/tf_unet) or [Pyramid
   Parsing](http://arxiv.org/abs/1612.01105), which seems to perform much better
   (but a working implementation is not yet available?). Run this scheme on test
   data from the *experiment generator*, using random selection as a first
   approach.
3. Improve on semantic segmentation with **instance segmentation** and/or
   **target identification**. This can take the output of a semantic
   segmentation, zero all background pixels, and determine an intelligent
   output.
4. Create the **imperfect oracle** with built-in random error emulating human
   labels, in
   [test_utils](https://github.com/bendkill/artifice/oracle/test_utils). This
   cannot be used until (4) is finished.
5. Develop an **active learning metric** for semantic segmentation. When using
   artifice, this would normally query the user. For our purposes, have it query
   the imporfect oracle. At first, this cannot include artificial images, since
   the oracle is unable to label those, and we haven't developed the image
   labeller yet.
6. Create the **performance evaluation** tool, to show network performance.
7. Improve the **experiment generator** to include anomalous behavior.
8. Develop different **data augmentation** methods for dealing with this
   anomalous behavior.
   
