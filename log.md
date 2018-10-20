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
artificial oracle for segmentation masks and other information, with noise added
to the oracle label as needed, to simulate human error.

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
