To be deleted.

This document is obsolete. Refer to design.md or the most recent entries in log.md.

# Perturbations

**Perturbations** are any variations between the sample images of an object and
actual experimental images. Scientific experiments may involve many
perturbations, some intentional and others unintentional. There are two primary
groups of perturbations:
* Measured
* Unmeasured

Measured perturbations result from the underlying target for prediction, usually
location. Possible targets include:
* Object location (in one or more dimensions, including **depth**)
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
* Global
* Local

We shall consider, in detail, perturbations of the following kind, either
unmeasured or unmeasured, both of which should be replicated in an Artificial
DataSet (ADS).
* Horizontal/vertical position
* Orientation (rotation, etc.)
* Horizontal/vertical scaling (distance from camera)
* Non-linear warping (such as on a folded paper)
* Partial occlusion (including specular highlights, dust motes, etc.)
* Object overlap
* Global illumination (uniform or nonuniform, ambient illumination)
* Local illumination (e.g. from a shadow) (blinn-phong, spotlights, directional
  lighting, omni lights?)
* Specular highlights (reflections)
* Structural variance between individuals (broad perturbation category requiring
  more information about objects being used.)
* Number of objects.
* Background (random noise, random images, etc.)

# Constraints

**Constraints** are limitations on the effects of perturbations. Universal
perturbations may have constraints as to the range of the perturbation, while
the constraints for isolated perturbations may be unique to each object. In
general, measured perturbations will have well-defined constraints, whereas
unmeasured perturbations should have more loose constraints, reflecting their
unanticipated nature.

# Experiment Design:

PovRay is a ray tracer, for generating synthetic images. Or OpenGL.

See also Vapory, a Python library for drawing POV-Ray stuff.

# Notes:

How do we quantify the fundamental Bas-relief ambiguity. Cannot care about depth
when light source is perturbable, fundamental problem.

Plan of attack:
* Make little stickers, markers (rotationally disambiguous)
* Get the synthesized images from computer graphics.
* Look at network design.
