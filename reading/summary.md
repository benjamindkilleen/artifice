# Perturbations

**Perturbations** are any variations between the sample images of an object and
actual experimental images. Scientific experiments may involve many
perturbations, some intentional and others unintentional. There are two primary
groups of perturbations:
* Measured
* Unmeasured

Measured perturbations result from the underlying target for prediction, usually
location. Possible targets include:
* Object location (in one or more dimensions)
* Object orientation (rotation on one or more axes)
* Number/type of objects

Unmeasured perturbations, on the other hand, result from unintended aspects of
the experiment, e.g. changes in lighting. Unmeasured perturbations usually
include:
* Lighting
* Object Shape
* Specular highlights
* Movement of extraneous objects

Perturbations may be further categorized as affecting either the whole of an
image or merely a single object in an image. These categories are referred to as
* Universal
* Isolated

We shall consider, in detail, perturbations of the following kind, either
unmeasured or unmeasured, both of which should be replicated in an Artificial
DataSet (ADS).
* Horizontal/vertical position
* Horizontal/vertical scaling (distance from camera)
* Non-linear warping (such as on a folded paper)
* Partial occlusion (including specular highlights, dust motes, etc.)
* Object overlap
* Universal illumination (uniform or nonuniform)
* Isolated illumination (e.g. from a shadow)
* Specular highlights (reflections)
* Structural variance between individuals (broad perturbation category requiring
  more information about objects being used.)

# Constraints

**Constraints** are limitations on the effects of perturbations. Universal
perturbations may have constraints as to the range of the perturbation, while
the constraints for isolated perturbations may be unique to each object. In
general, measured perturbations will have well-defined constraints, whereas
unmeasured perturbations should have more loose constraints, reflecting their
unanticipated nature.
* 

