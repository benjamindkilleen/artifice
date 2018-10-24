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
