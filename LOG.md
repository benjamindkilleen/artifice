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

## June 18, 2019: Summer Thoughts

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

## June 24, 2019: Reworking Artifice

Ideas in Artifice can be built on, code as well. Big ideas:
* Separate thread for querying. Make continuous queries and update the data that
  networks learn from. Can simulate human queries with a simple delay.
  * Got rid of query size.
  * Got rid of num-candidates.
  * This makes the querying process TOTALLY separate from training. All training
    has to do is reload data from the data-dir every epoch, since more examples
    may exist in the annotated set. Doesn't even have to know the size of the
    annotated set? Just the epoch size.
  * Got rid of subset size. May have to put it back.
* Instead of `splits` as the way to divide data, we now have `data-size`, which
  is the size of all the data, and `test-size` which designates how many
  examples to withhold for testing. This is because most applications won't be
  doing testing. They'll just care about 
* Make the default hourglass net 3 layers deep, not 4.

## July 3, 2019: Annotator
* Can make the unlabeled set, skipping test set as necessary of course.
* Labeled set maker modified to take as many images as there exist labels for,
  assuming the same ordering. Not doing anything to match filenames.
* Put human labeled/annotated sets in directories, since there will necessarily
  be a lot of these.
* The annotator runs at the same time as the trainer.
* model.train method 

## July 8, 2019: Augmentation

* Added a simple augmentation to randomly translate images. Without
  augmentation, an epoch (2000 images) took 490s. After this simple augmentation
  method, it took 588s, not including shuffle time. So not bad.
* Computer isn't getting hot though, suggesting that augmentation is a serious
  bottleneck.
* Caching just doesn't work. Need to reach the end of the iterator before it
  actually saves the cache, so that's a thing. Gotta figure that out.

## July 10, 2019
* Changing to just translating patches, training with an augmented set took
  527s. This is a marginal improvement over 588s.
* One problem with reloading the dataset every epoch like we're doing is having
  to fill up the shuffle buffer. Is this worth it?
* Could detect when the annotator is done, and then just call fit up till
  epochs? That would be good.

## July 13, 2019: Experimenting with Caching
* Seeing how long each epoch takes with caching enabled.
* Had to take `steps_per_epoch` from the dataset, since it had already been
  batched. May want to take `size` before batching, so that shuffling and
  batching can be done after caching.
* Why is the shuffle buffer not being filled anymore?

## July 15, 2019: Running 3 epochs on GPU
* eager, no cache: 40, 24, 24s
* patient, no cache: 43, 28, 28s <--- yielded lowest training error after 3 epochs
* eager, cache: 31, 21, 21s <--- best run, by time
* patient, cache: 35, 25, 25s

So pretty clearly, actually, running with eager execution and caching is the way
to go.

## July 19, 2019

Todo today:
* See if there's any possibility of or advantage to grabbing windows around
  registered objects at each scale and preserving pixels around it.
* Note that this requires a peak-detection layer. Custom keras layer maybe?
  Returns shape `(batch_size, num_objects, 2)` int64 tensors giving positions of local peaks?
* 

Notes on sbnet:
* The nvcc library is here: /software/cuda-9.0-el7-x86_64/bin. Add that to path.
* Need to change the path that Makefile looks for cuda/include.
* Found tensorflow at: 
  /software/Anaconda3-2018.12-el7-x86_64/envs/tf-gpu-1.13.1/include
* Added the `-D_GLIBCXX_USE_CXX11_ABI=0` option to both 

## July 22, 2019
* Implemented the sparse conv2d stuff, now I'm running into this cryptic error:
```
tensorflow.python.framework.errors_impl.InternalError: Could not find valid device for node.
Node: {{node ReduceMask}}
All kernels registered for op ReduceMask :
  device='GPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_FLOAT]
 [Op:ReduceMask]
```
  which is just interesting. Maybe it needs to be run without eager execution, I
  dunno. I'll figure it out later.
  
## July 23, 2019
* Maybe we should revert ProxyUnet to just predict the last layer, or have three
  models, one that does just the last level, one that does every level but not
  sparsely, and finally SparseUNet.
* TODO: change dat.py so that the different between data subclasses is not the
  structure of the files from which they read, which can be set at instantiation
  just by passing in a parse/serialize function, but rather in the process
  function, which can design different output for different models.
* This doesn't matter so much for ProxyUNet vs SparseUNet, which take the same
  style of outputs, so can keep it the same for now.
* But then again, maybe the model should be able to select the different kind of
  outputs. We'll see.
* Currently needing to implement sparse tranpose convolution as well as sparse
  upsample for the mask?

## July 24, 2019: Timing the sparse net
I used the 500x500 disk images with 10000 images, batch size of 4, on GPU. (No
caching.) I ran 3 epochs, no augmentation or anything. Fully labeled.
* ProxyUNet: 324, 268, 258s
* SparseUNet: (didn't get to work)

## July 25, 2019: Debugging gradients for sbnet
The plan:
* implement `sparse` package in artifice, which imports from sbnet, more
  tightly. Make sure gradients are working, which was the problem. I never
  registered gradients.
* In `lay`, make sure each layer is totally working by building a simple model,
  just the identity function is fine. After that, should be good to go.

## July 26, 2019: Timing the sparse net (actual)
I used the 500x500 disk images with 10000 images, batch size of 4, on GPU. (No
caching.) I ran 3 epochs, no augmentation or anything. Fully labeled.
* eager ProxyUNet: 
* eager SparseUNet: 
* patient ProxyUNet: 
* patient SparseUNet: 372, 262, 256

Changing the `block_length` for interleave didn't have much effect. Still
hitting bottlenecks:
* patient SparseUNet, double block_length: 380s

There are some notable bottlenecks which I'm not sure where they could be coming
from. Might want to tweak the data generation settings? Also need to consider
using variables for sparse scatter.

Todo, in no particular order
* migrating weights between sparse/dense UNet implementations. In theory, if we
  load by weight name, these should be compatible, but we have to make sure that
  using different layers doesn't result in different weight names. Could be a
  headache.
* CPU implementations of reduce_mask, etc. Can largely grab implementations from
  `sparse_conv_libs.py` directly. Should just make prediction easier.
* Multiscale tracking code, using the output at each level.
* Add Variable to Sparse Conv layer implementations. Each layer would just have
  a variable that stores its outputs that gets set to 0 once, during each pass,
  as opposed to copying the same tensor a bunch.

## July 29, 2019:
Multiscale tracking is still messed up. Figure out why.

## July 30, 2019: Figured out multiscale
For some reason, multiscale tracking (in post-eval analysis), as implemented
with `peak_local_max` and the `labels` keyword actually takes longer. The
difference is stark. On the unlabeled disk set, which has 10,000 images, batch
size of 4:
* without multiscale tracking: 34.2s
* with multiscale tracking: 44.2s 
* sparse, no multiscale: 36.3s
* sparse, no multiscale, patient: 34.1s

Note that both options still used the multi-output UNet. Still, this is not
encouraging.

For training, we observed a slight speedup using sparse evaluation (patient
execution):
* dense: 368, 288, 286s
* sparse: 386, 269, 255s
* sparse, use-var: 401, 279, 279s
* dense, cached: 150, 136, 136
* sparse, cached: 153, 117, 116
* sparse, use-var, cached: 345, 138, 137s 

So sparse, cached exhibited a 15% speedup over dense, cached, as opposed to a
10% speedup without caching. Use-var showed no speedup, possibly because of the
overhead of initializing/accessing variables?

I'm noticing that this is hanging up a lot on something, so I'm wondering if the
eval is actually not the bottleneck.

Possible update that would make things faster:
* The gradient for `scatter` doesn't use `sparse_scatter_var`, so maybe that's
  another bottleneck? Probably not though.

## July 31, 2019: Rerunning with more levels
We using patient execution, with level filters: 128, 128, 64, 32. Input tile
shape, (on gpu), 10000 500x500 images with batch_size of 4.
* dense, no cache: 1289 (~18m), 1034, 1021s
* dense, cache: 1062, 459, 459s
* sparse, cache: 547, 401, 396s

which is like a 13% speed boost, ish, when cached.

## August 1, 2019: Idea: 
Create layers for the `sparse_gather` and `sparse_scatter` operations so that we
don't scatter unless absolutely necessary, potentially just at the end of the
first layer and at the end of the last layer. This lets us use fewer of the
outputs for the loss (potentially cutting out that data processing/loading
bottleneck a little bit,) but also requires some fancy arithmetic on the blocks.

For now:
* Re-implement `reduce_mask1`, `gather`, `scatter`, and `scatter_var` for CPU.
* Look into benchmarks for evaluation.
* Think about fully-sparse U-Net `FullSparseUNet`.

### More time checks:
I rebuilt the sparse ops with tf primitives. Here's how they run on GPU, using
caching and patient executiong. 

Just as a reminder, the sbnet runtimes were (pre-cached, patient, gpu):
* dense: 150, 136, 136s
* sparse: 153, 117, 116s
* sparse, use-var: 345, 138, 137s 

And my implementation with tf primitives:
* sparse: 321, 143, 142s (not pre-cached)
  pre-cached: 175, 148, 144 (comparable)

Use-var resulted in a gradient error, probably due to `tf.scatter_nd_update`.

Okay, so tf primitives are not unreasonable, but not as good as the sparse,
non-variable option, and use-var was actually better. CPU evaluation worked as
expected.

Messing with the batch size, using sparse, gpu, patient, caching:
* -b 8: 354, 93, 92s (not pre-cached)
* -b 16: 383, 92, 83s

so batch size 8 seems to be the way to go, since it's a huge improvement over
-b 4. 16 is also fine, might improve on 8, so we'll go with that from now
on. Higher batch sizes wouldn't evenly divide the 10000 size dataset, so
whatever.

The full training run, using these settings:
* dense: 100, 91, 91, 91, ...
* sparse: 212, 91, 83, 81, 80, 80, 79, ...

The dense network never dropped between 91s, but the sparse network got
below 80. 

## August 5, 2019:
Docs for continuity:
* aspects familiar for tensorflow developers
* aspects unfamiliar for tensorflow developers
* usage examples
* breadcrumbs

Document "wisdom"/"bread gained through experiment:
* how I go about debugging tensorflow
* different speedups
* intuition about model topology

More speed tests:
* The sparsity tolerance vs accuracy.
* Learn the tolerance
* learn the sparsity hyperparameters?
* "I want you to be sparse, you figure out how."
* Effect of final max-peak search learned?

## August 6, 2019:
Speed test results. As a reminder, patient, batch size ]:
* dense: 95, 86, 86
* sparse: 321, 97, 81,
* better sparse: 144, 109, 108
  142, 109, 107 whyyyyyy

## August 7, 2019:
* glk: "Model has to kind of find itself."
* change the threshold across time/levels?
* pursue H_G

## August 8, 2019: More speed tests:
Updated better-sparse so it actually learns:
* dense: 95, 86, 86 (from before)
* sparse: 321, 97, 81, (from before)
* better sparse: 86, 77, 77, (break), 86, 77, 77, 77, (break), 86, 77, 77

Another few runs were even faster:
* 84, 75, 75, ...
* 98, 65, 64, ...

That's like a 30% speedup over dense UNet (91s/epoch), theoretically

But the pose image loss doesn't seem to be going down a whole lot. Maybe I need
to reprioritize the `pose_image_loss` over the sparsity losses?

The auto-sparse unet has losses (without loss weighting):
* pose_image: 0.026, 0.0250, 0.0246, 0.0243, 0.0241, 0.0239, 0.023, 0.0236,
  0.0234, 0.0233
  
The output looks better after reprioritizing the `pose_image`. Maybe now make
self.tol lower? Also, we should try dropping `self.tol` as a function of
epoch. Not sure how to do that at all.

### Evaluating different models after 20 epochs:
Tol set at 0.05:

UNet:
```
INFO:artifice:objects detected: 36396 / 40000
INFO:artifice:avg (euclidean) detection error: 2.9550125168617627
INFO:artifice:avg (absolute) pose error: [0.04189316 0.75587162]
INFO:artifice:note: some objects may be occluded, making detection impossible
INFO:artifice:avg: [2.95501252 0.04189316 0.75587162]
INFO:artifice:std: [2.10017321 0.02714674 1.0314036 ]
INFO:artifice:min: [9.02138427e-02 9.45106149e-06 1.09195709e-04]
INFO:artifice:max: [9.98716831 0.13081706 7.21399355]
```

Sparse:
```
INFO:artifice:objects detected: 36420 / 40000
INFO:artifice:avg (euclidean) detection error: 4.4020743376619755
INFO:artifice:avg (absolute) pose error: [0.05856278 1.71808761]
INFO:artifice:note: some objects may be occluded, making detection impossible
INFO:artifice:avg: [4.40207434 0.05856278 1.71808761]
INFO:artifice:std: [2.65426281 0.04823871 1.20276581]
INFO:artifice:min: [0.08930295 0.00054923 0.00357652]
INFO:artifice:max: [9.9862299  0.29404551 5.14802027]
```

Better-sparse:

