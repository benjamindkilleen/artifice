# Log

## October 20, 2018

### Goal
To create an ML workflow for image analysis in laboratory
experiments.

Following advice from Risi Kondor and Michael Maire, I've decided to pursue a
double-stack CNN architecture, using semantic segmentation to first identify
objects of interest, then a more simple CNN to extract final target features, if
needed. This allows for the most flexibility with regard to desired target
features, marker shape, etc. It also allows for advanced dataset augmentation,
which may prove vital especially with regard to scientific images.

### Scientific Method Problem
Unlike ML workflows for real-world systems (e.g. self-driving), scientific
experiments generally aim to capture an underlying structure to the problem. A
ball thrown in the air, for example, will follow a path pre-determined by the
laws of gravity. However, a model trained only on images from this experiment
will reflect that underlying structure and fail to recognize anomalies as a
human might. This is a fundamental problem relating to the scientific method
itself: your hypothesis space cannot influence your measurement. If, once every
thousand experiments, the laws of gravity caused a ball to fly differently, an
ML model would be ill-prepared to recognize it.

### Relevant Literature
As such, we aim to employ data-augmentation using the learned segmentation masks
during an active learning process. Relevant literature includes:
* Active learning generally: [Active Learning Literature
  Survey](http://burrsettles.com/pub/settles.activelearning.pdf)
* Active learning with semantic segmentation: [Active Learning for semantic
  segmentation with expected
  change](https://ieeexplore.ieee.org/ielx5/6235193/6247647/06248050.pdf?tp=&arnumber=6248050&isnumber=6247647&tag=1)
* Active learning for bounding boxes: [Localization-Aware Active Learning for
  Object Detection](http://www.arxiv.org/pdf/1801.05124.pdf)
* Segmentation architecture: [U-Net](http://www.arxiv.org/pdf/1505.04597.pdf)

### Experimental Design
Using the POV-Ray tool paired with [Vapory](https://github.com/Zulko/vapory), I
can create arbitrary experimental scenes, with known object target information,
in order to measure the performance of the model. This can also serve as an
artificial oracle (edit: "imperfect oracle") for segmentation masks and other
information, with noise added to the oracle label as needed, to simulate human
error.

This dataset should be constructed to show the model's resilience to
anomalies. That is, the model should not anticipate target features in any way,
and the model should be able to understand unfamiliar scenes in the test set.

Obviously I hope to not have to write my own image labeller for semantic
segmentation and other target data. The novel approach is the combination of the
segmentation net with a secondary CNN, also trained with active learning. This
second net should require much less training, since it ideally only acts on the
isolated objects to extract target features (image-space position, orientation,
etc.)
Possible labellers include:
* [LabelBox](https://github.com/Labelbox/Labelbox) has a 5000 image limit on unpaid
  version, but it exports to TFRecord. However, it doesn't seem built for active
  learning.
* [LabelImg](https://github.com/tzutalin/labelImg) is open-source and runs
  locally, but seems to only support bounding boxes. It is python-based, though,
  and could be modified?
* [OpenLabeling](https://github.com/Cartucho/OpenLabeling) is also open-source,
  and potentially more advanced, but also limited bounding boxes with YOLO
  specifically in mind.
* [PixelAnnotationTool](https://github.com/abreheret/PixelAnnotationTool) from
  abreheret is a Qt-based segmentation tool, using a drawing tool rather than a
  polygon labeller. It is free but cannot be changed. **Runs locally.**
* [semantic-segmentation-editor](https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor)
  is also an open-source segmentation tool. It seems slightly less
  well-developed, but runs locally. **Might be limited to Windows.**
  
### Ongoing Questions:
* **Should each object be of a unique class?** For experiments with potential
  occlusion, this seems desirable. For experiments where occlusion is
  impossible, it shouldn't be necessary.
* **Should the user expect the model to eventually be fully independent, or will
  anomalous data always require active learning?** This is a response to the
  hypothesis-space problem.

## October 21, 2018

### Misc
* A tensorflow implementation for the U-Net architecture can be found
  [here](https://github.com/jakeret/tf_unet), originally used for
  [this](https://arxiv.org/abs/1609.09077) paper on galaxy detection.
* See [docs/design.md](docs/design.md) for an ongoing design document for
  artifice.
* Additional target features could include some parameterization of the
  object. Need the ability to add arbitrary numerical or categorical targets, as
  long as consistent.
* The
  [deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)
  repo is a TensorFlow supported segmentation model.
* Potential problem: if we ever want to show the user artifical images, as part
  of the selection metric, then the imperfect oracle cannot be used. These cases
  will have to defer to the human labeller.
* We need to provide support for *existing data*. Very often, a scientist will
  already have well-labeled experiments, and this will provide a useful starting
  point for any experiment.

### Experiment Generator
The **experiment generator** needs to store its output data in a form readable
by [tf_unet](https://github.com/jakeret/tf_unet), i.e. it needs to be in a
tfrecord. But each record needs to include both the mask and the 

### Literature
* [ParseNet: Looking Wider to See Better](https://arxiv.org/abs/1506.04579)
  explores methodology for better segmentation masks.
* [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected
  CRFs](https://arxiv.org/abs/1412.7062) made huge strides in semantic segmenation.

### Experimental Design: artificial experiments.
There are numerous experiments we could try to emulate, each with their own
target features.
* **Find the ball:** the simplest experiment, featuring a single ball as the
  marker. The ball's center can determine position. Added complexities include:
  * Constraining ball position according to some distribution.
  * Moving ball in +/-z.
  * Changing number of balls, possibly requiring uniqueness for each one, if
    occlusion is possible.
  * Adding orientation markers to the balls.
  * Perturbing the background image.
* **Petri dish:** simulate the inside of a petri dish, finding the mask for an
  "amoeba." In reality, this can be some shape with sinusoidal distortion of the
  edges.

## October 22, 2018
* See [this example](https://gist.github.com/Zulko/f828b38421dfbee59daf) on
  using vapory for realistic physics simulations. This would be cool for doing a
  bouncing ball experiment, without having to explicitly set each location, etc.

## October 23, 2018
* Potential problem where POV-Ray and tensorflow are mutually exclusive. Might
  need to add functionality for experiments.py to export to `png` or `json` for
  each image. This would allow the script to create datasets from povray in one
  step, then convert them to tfrecords in the next step. This is clunky,
  however.

## October 24, 2018
* RCC responded to the above issue. POV-Ray depends on TIFF, which hadn't been
  updated for gpu2.
* New issue for experiments.py: target shouldn't just be one vector. In a sense,
  every object being tracked has a target vector associated with it. Associating
  *target vectors with regions in a segmentation mask* is tricky. Requires more
  thought, because the targets are only being used for the second-step net.
  
### Target Association Problem
**Problem:** we need to associate with each instance of an object in the image a
specific vector, which is the target for that object. The problem is that,
while forward inference after training should work fine, there's no way to know
which target vector is associated with which object instance in the image.
* We could make each instance it's own class, in the dataset. This is clunky and
  doesn't reflect the structure of the problem, however.
* Look into how object detectors deal with unknown numbers of objects (use
  **LSTMs with a confidence metric**), how they deal with backpropagation in
  this case. Should be the same deal. Maybe Prof. Maire can offer reading? This
  solution ties together the two parts of the network approach, making the
  second step less isolated.
* Have an unambiguous ordering for each target vector. For a known number of
  objects in the scene, this can be the (x,y) coordinate, lexicographically
  ordered? Still some ambiguit, and object position is not necessarily
  well-defined from a mask, hence the problem.
* Include position in every target. Find an unambiguous mapping from every
  detection to every target.

### Generating Masks:
* Need to **determine camera transformation matrix** from POV-Ray setup. Use
  this to generate masks, for each type of vapory object, manually? Should be
  relatively simple...

## October 25, 2018
* Interesting problem of semantic segmentation. First step, need to deduce the
  WtoI camera projection matrix of the POV-Ray camera, relatively easy. Second
  step: Experiment compute_mask needs to check every overlapping pixel for the
  closer object. It does this by first getting all the masks that lie on a given
  pixel, then calculating (based on the geometry of each object) which one is
  closest. This isn't the fastest thing in the world...maybe compile to cython
  (without modifications)? Gross. Maybe write a C library to do it for each
  object? Hmmmmmmm. More gross. Staying in native python isn't bad either, if we
  just run this on midway.
  Problem is, we're gonna have to stick with datasets on spheres, boxes, and
  maybe cylinders. Just for the test stuff.
* Another issue: perhaps we shouldn't be creating the complete dataset? Or
  rather, it seems like the oracle will have to take the complete dataset and
  only show artifice a small part of it.

## November 5, 2018
As of now, I've completed the initial work on experiment.py, a tool for creating
semantic segmentation datasets with POV-Ray. The next step is to create the core
ML functionality for training and architecture. This requires some level of
generality, allowing for an active learning metric to replace the random
selection default. As discussed, the implementations of
[tf_unet](https://github.com/jakeret/tf_unet) are a good place to start.

Initially, of course, simple semantic segmentation is a good place to
start. Eventually, though, we want to incorporate the desired targets into the
annotation. This may involve "distance to object center" as a good localization
metric (even if an actual center isn't well defined, can be an average). This is
good for localizing any number of objects, potentially.

There's also the graph partitioning method (to read).

## November 6, 2018

### Notes on Deep Watershed Transform
* Deep network learns an intermediate task: direction of descent of watershed
  energy.
* Takes a semantic segmentation as input. Can use any algorithm for this initial
  semantic segmentation. Input 0s all the pixels in the background class, then
  feeds in the remaining RGB image.

## November 8, 2018

### Planning Artifice Layout

The structural elements of artifice are laid out in
[docs/design.md](https://github.com/bendkill/artifice/blob/master/docs/design.md),
but here I give some thought to the layout of the code and how each piece will
interact with each other.

Note that this diagram is mutable and subject to change. Consult
[docs/design.md](https://github.com/bendkill/artifice/blob/master/docs/design.md)
for a more up-to-date summary.

```
artifice
├── README.md
├── artifice
│   ├── semantic_model.py
│   │   """Provides an abstraction of the tensorflow model used for semantic
│   │   segmentation, most likely tf_unet.
│   │   """
│   ├── instance_model.py
│   │   """Provides an abstraction of the tensorflow model used for instance
│   │   segmentation after semantic segmentation blackout. Most likely an 
│   │   implementation of Deep Watershed Transform. This could potentially
│   │   also include target annotations on each instance, once identified."""
│   ├── label.py
│   │   """Queried with an example from a dataset. Returns existing annotation,
│   │   annotation from imperfect oracle, or human annotation, depending on
│   │   stage of dev."""
│   ├── augment.py
│   │   """Provides dataset augmentation capabilities, given instance segmentation
│   │   annotations and images. Meant to produce images for first-input (before
│   │   semantic segmentation blackout.)"""
│   └── utils
│       └── dataset.py
├── docs
│   └── design.md
├── log.md
├── scripts
│   │   """Miscellaneous python scripts, mostly for dataset generation
│   │   """
│   └── two_spheres.py
└── test_utils
    │  """Contains tools for testing artifice which are not part of the core
    │  functionality. e.g. dataset generation (datasets normally provided by user), 
    │  or imperfect oracle (normally the user serves as the "human oracle")""".
    ├── imperfect_oracle.py
    │  """Provides an artificial human oracle. Given an annotation, return an
    │  imperfect annotation, such as a human might produce."""
    ├── draw.py
    │  """Emulation of skimage.draw, providing other shapes as needed. Not
    │  currently in use."""
    └── experiment.py
       """dataset generation"""
```

Depending on obsoleteness, this section may also appear in
[docs/design.md](https://github.com/bendkill/artifice/blob/master/docs/design.md).

### Misc
[tks10/segmentation_unet](https://github.com/tks10/segmentation_unet) is another
very good implementation of UNet, using
tensorflow. [tf_unet](https://github.com/jakeret/tf_unet) still seems like a
better implementation overall (more robust model setup), but `segmentation_unet`
has a data augmenter which may be good as a starting point for our augmentation
setup

## November 13, 2018
Check-in meeting notes:
* Balls moving around inside some cube, bouncing off one another.
* Elastic, inelastic collision?
* Spheres coupled with an invisible spring, learning the spring constant through
  the ML recovered positions. How would you make sure that the recovery of one
  ball's wouldn't influence the other's.
* Initially just learn on two spheres.
  * Spring constant, different masses, etc.

## November 14, 2018
For a more awesome approach to `mp4` writing, see
[this](https://github.com/Zulko/moviepy/blob/master/moviepy/video/io/ffmpeg_writer.py) great repo. It's a wrapper around ffmpeg.

## November 29, 2018

### Meeting Notes:
* Better terms for imposed/intentional constraints.
* When writing a paper, give reader warm fuzzy feelings. Risk: superficial
  appearance of inventing new words/formalisms that the world has already
  figured out how to describe. Search for high-level conceptual modeling of a
  standard computer vision task.
* People don't name papers. So remove "Artifice" from report.
* Need a figure that encompasses what Bias is in an ML context, what is to us,
  what the difference is, etc.
* Adverse noise: REDUNDANT.
* Relate imposed constraints and inherent constraints to the things that span
  the tangent space.
* "Pencils don't move sideways." Gordon Kindlmann, November 29, 2018
* "Experimenter's constraints"
* Should noise affect the augmenter? We have no parameterization of it, but 
* Distinction between what the DNN learns to cope with and what the augmenter
  helps it learn.

### Terminology:
* Bias: a tendency in the dataset (sample) that is not reflective of the world
  (Omega), but also bias of an esimator? Bad word. 
* IID: independently and identically distributed random variable
* Noise: degrees of freedom you don't know how to parameterize?

## December 7, 2018
### Further Discussions of Terminology
* **example**: x, an MxN image, or 1x(MxN) vector.
* **example space**: vector space of which examples are elements. For MxN
  images, this is R^(MxN)
* **dataset**: X, consisting of many examples. We impose no ordering on
  examples.
* **distribution**: (f : R^(MxN) -> R) probability (density) of an example
  x. There are several notions of distributions:
  * **natural distribution**: the real-world distribution from which original
    data obtained.
  * (unnamed) notion of *uniform distribution*: an imaginary distribution (which
    an augmented dataset simulates being drawn from)
    * *uniform distribution*
    * *something-agnostic distribution*?
    * *naive distribution*
    * *chaotic distribution*
    * *nature-agnostic distribution*
    * *confined distribution*
    * **agnostic distribution**
  * The **agnostic distribution** is uniform within the *absolute boundaries*
    and "agnostic" to *governing trends*.
* (unnamed) notion of constraints imposed by the experimenter: these *well-parameterized*
  outer bounds of the data space are *known* by the experimenter. They consist
  of *boundaries* and do not govern the density function of the data within the
  space. (i.e. the data need not fill out its space)
  * *boundary constraints*: the known parameters constraining the "natural
    distribution."
  * *distribution boundaries*
  * *imposed boundaries*
  * *artificial boundaries*
  * *absolute boundaries*
  * *constraints*
  * *experimental constraints*
  * *absolute constraints*
  * *data constraints*
  * *data space constraints*
  * *boundary parameters*
  * **label-space boundaries**
* (unnamed) notion of the space these constraints/boundaries define:
  * **data region**
  * *bounded region*
  * *data neighborhood* (nope, too much a notion of locality)
  * *data space*
* (unnamed) notion of constraints resulting from object of study: these are not
  known with certainty and affect the *natural distribution*.
  * *object principle*
  * *target principle*
  * **governing principle**
  * *target structure*
  * *hypothesis structure*
  * *hypothesis factor*
  * *principle distribution*
  * *object of study*
  * *non-uniform term*
  * *heterogeneous term*
  * *structured term*
  * *structured patterns*
  * *object patterns*
  * *natural patterns*
  * *governing patterns*
  * *underlying patterns*
  * *enigmatic patterns*
  * *non-uniform trends*
  * *heterogeneous trends*
  * **governing trends**
  * *natural structure*

## December 9, 2018
For the coupled springs experiment, maybe apply different walls for each
object. These walls constitute known *boundaries*, within which the experiment
produces a sample from the natural distribution, but you want to sample
uniformly within chi.


## January 9, 2019

Feedback on Report:
* Related works section is nice, summarizing all the related works, etc. Where I
  believe I am. Here are the other nearest people, the ways they've tried to do
  similar or related things. Performance of academic mastery. This is a "symptom
  of mature work."
* Bad form to use the citation as a subject in the sentence.
* Some people use a macro for "et al" to make sure it's not treated as end of
  sentence, not broken up over two lines.
* Every equation needs to be numbered.
* Forgot a backslash with "mathbb".
* More schematic figures that unify the mathy things you're talking
  about. Figures always enhance understanding.
* When you have a new variable, best if you have some way of having it appear
  first in a numbered equation OR have it appear in some table that defines
  variables and equations.
* Paint in broad strokes what the query strategy is for. Fill in someone who has
  no idea what active learning is what the query strategy is doing. What are the
  parts/outcomes of active learning that I'm drawing on.
* More schematic diagrams that use this idea that distribution is what matters,
  and we don't want to recognize it. Visual ways to illustrate that.
* For illustrative purposes, the shadows in the example.
* Annotation space is not in that figure.

## March 23, 2019: Design Notes

* There are two types of data, fundamentally. Those with semantic annotations,
  useful for transforming, augmentation, and those without. We create a training
  set from the semantically annotated data.
* From experiments, we generate the images as PNGs, the labels as a single .npy
  file, and the semantic annotations as PNGs as well. 

## April 1, 2019: Batch results

Batching the transformation operations resulted in much faster epochs. Now, a
single epoch takes about 4-5 minutes, whereas without batched transformations, a
single epoch could take about 24 minutes. This approximately 4x speedup
correlates with the (untiled) batch size of 4, showing that the majority of time
is spent generating the actual examples.

The training exhibits this really strange behavior where the first few examples
take quite a long time, possibly to build the computation graph, after which
batches of 16 examples (weirdly enough) are processed very quickly before a
slight pause. Unknown if changing the batch size affects this behavior, but it
shows that the GPU is being underutilized during data generation on the CPU.

Still much quicker than unbatched data generation.

## April 3, 2019: Oops

The above speedup may have been due to a `steps_per_epoch` error. Additionally,
it relied on batches of `batch_size` tiles, when we really want
`batch_size*num_tiles` tiles per batch, in order that they can be conveniently
reconstructed. Now, it seems, and epoch should take about an hour. So was there
any speedup at all? Possibly not. Might want to go back to the previous
paradigm of batching after augmenting. Possibly. 

## April 4, 2019: Query Selection for Augmentation

Idea: after training for a bit on one example, maybe select the next example
based on KL divergence of the predicted field with an expected field where
object locations match the predicted max peaks. Could also consider methods
based on the activations of the peak in question? Like, somewhere, the peak
should have a value greater than 1. If the peaks don't have very high values,
then maybe that's an uncertain prediction.

Of course, the KL divergence idea also tests this. Be sure to use the same
tiling function.

### Results:

Training for 20 epochs, using skimage local peak finder, yields:
```
INFO:artifice:average error: 2.25
INFO:artifice:error std: 1.94
INFO:artifice:minimum error: 0.04
INFO:artifice:maximum error: 8.71
```

## April 5, 2019: Results:
Training for 50 epochs yielded:
```
INFO:artifice:average error: 2.07
INFO:artifice:error std: 1.86
INFO:artifice:minimum error: 0.02
INFO:artifice:maximum error: 9.01
```
Which is better, actually, but still not great.

## April 7, 2019: Results
After training for a few epochs with active learning approach, we get
```
INFO:artifice:average error: 2.04
INFO:artifice:error std: 1.47
INFO:artifice:minimum error: 0.02
INFO:artifice:maximum error: 7.63
```
marginally better BUT the error is now seemingly more random, due to edge
effects and whatnot. Crucially, we don't observe the shadow dependent behavior,
and we also see much better performance at the edges of the whole image. Problem
seems to arise from untiling. Perhaps further training will improve, still very
preliminary.

Need to do active learning for a certain number of epochs, then continue on with
regular learning. Perhaps the learner.fit method can return the active
`annotated_set` with `annotated_size` (after that many epochs), and then
training can continue to fill out the difference.

Also, be sure to not keep updating the background image? Something to consider.

### Thoughts on padding:

Reflection padding is resulting in possible better edge performance, but it
could also be hurting the performance in those regions slightly. Something to consider.

## April 7, 2019: Results from active learning
After 8 epochs (10000 examples each), we have
```
INFO:artifice:average error: 1.91
INFO:artifice:error std: 1.35
INFO:artifice:minimum error: 0.02
INFO:artifice:maximum error: 6.13
```
which is the best so far.

## April 9, 2019: using Batch Normalization
```
INFO:artifice:average error: 1.61
INFO:artifice:error std: 1.39
INFO:artifice:minimum error: 0.02
INFO:artifice:maximum error: 7.28
```
which has lower average error. This was after just 8 epochs(failed because of
error, now caught). So batch normalization actually does help.

## April 11, 2019: results from active learning, larger tiles
After 12 epochs:
```
INFO:artifice:average error: 1.69
INFO:artifice:error std: 1.20
INFO:artifice:minimum error: 0.03
INFO:artifice:maximum error: 7.43
```
about the same as above.

## April 16, 2019: Results from labeled, augmented, and learned methods.

Each trained for 16 hours, but the labeled training was somewhat quicker. In
that time, it managed to train for 18 epochs, while augmented trained for 13,
and learned for 12. 

For "labeled":
```
INFO:artifice:average error: 100.06
INFO:artifice:error std: 69.18
INFO:artifice:minimum error: 0.26
INFO:artifice:maximum error: 383.96
```

For "augmented":
```
INFO:artifice:average error: 102.16
INFO:artifice:error std: 71.51
INFO:artifice:minimum error: 0.28
INFO:artifice:maximum error: 418.25
```

For "learned":
```
INFO:artifice:average error: 100.21
INFO:artifice:error std: 70.95
INFO:artifice:minimum error: 0.52
INFO:artifice:maximum error: 430.97
```

This is obviously a bug that's been introduced, possibly with the new fielding
strategy? (1/r^2 as opposed to 1/r).

Looking at the video results, it's obvious that the error here is not with the
actual model but rather the untiling process. Obviously a bug introduced with
that. Still figuring it out. But the good news is, choosing the sharper field
seems to result in actually sharper outputs.

### Fixed

Oh, I'm dumb. I was shuffling the test inputs, so of course it
failed. Spectacularly. After fixing this bug, we have:


For "labeled":
```
INFO:artifice:average error: 0.49
INFO:artifice:error std: 0.43
INFO:artifice:minimum error: 0.02
INFO:artifice:maximum error: 6.26
```

For "augmented":
```
INFO:artifice:average error: 1.61
INFO:artifice:error std: 1.35
INFO:artifice:minimum error: 0.03
INFO:artifice:maximum error: 8.70
```

For "learned":
```
INFO:artifice:average error: 1.27
INFO:artifice:error std: 0.98
INFO:artifice:minimum error: 0.03
INFO:artifice:maximum error: 5.06
```

Some interpretation: the main source of improved accuracy, I would guess, is the
change in the field function. This just encourages sharper peaks around the
center, making life easier for the classical peak-finding algorithm. This is
interesting becuase it's a combination of modern deep networks with classical
image analysis. 

Additionally, it should be noted that the "labeled" set had quite a few more
epochs to train. Of course, its relative accuracy is not a huge source of
concern, since we achieve comparable performance with much fewer human
annotation. This is the chicken and the egg problem: goal is to analyze the data
quickly, but to train a model to analyze it, we already need analyzed
data. Augmentation solves this somewhat, active learning even better.

Just for kicks, going to knock off the last few epochs of training, bring the
labeled set back to twelve, and see how it does.

Notably, the "learned" stratgy has a lower maximum error, but the std isn't as
good as labeled.

### June 18, 2019: Summer Thoughts

* Probing or Point of Interest (PoI) network emulates Fast R-CNN by taking in a
  series of predictions for objects that all rely on the same feature vector.
* Then, after making a feature map for the entire image, feeds that feature map
  into K regression nets that output a D-dim tuple for however many
  outputs. 
* Other outputs don't matter, as long as they can be learned regressively (in
  range [0,1]), but we need to output the likelihood of detection (really a
  classification, between 0 and 1 one hot), predicted position, and other info.
* Training consists of taking random PoIs around known objects (in the gyro or
  other data).
* Following Girshik (Fast R-CNN), take 25% of PoIs from around known objects,
  with a gaussian distribution, with prob 1 and the true offset. Set a distance
  threshold to ensure no GT is too far away from a real object.
* Sample the remaining 75% of points from background regions, greater than
  DIST_THRESH from any object.
  
Details for coding:
* The stored data needs to include:
  * Image
  * List of object labels, as before. That's it?
* So we can take the predictions from the trained network. Totally.

New repo: `probal`.
