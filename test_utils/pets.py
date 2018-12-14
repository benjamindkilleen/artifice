"""Create tfrecord files from the pets dataset.

For the sake of this example, we ignore the different breeds of dogs and cats,
instead mapping every example to one of two classes, either dog or cat. We
consider the boundary-region of each trimap to be part of the background, for
simplicity.

So our class mapping is:
{ 0 : 'background',
  1 : 'dog',
  2 : 'cat' }

But the original trimaps have the following mapping:
{ 1 : 'animal',
  2 : 'background',
  3 : 'edge' }

Note: some of this implementation is based on
https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pet_tf_record.py,
which deals with this dataset directly.

In this script, we produce increasing levels of data scarcity, as well as simple
augmentation methods, per image. These are controlled via the command line.

Should be run from $ARTIFICE, as always.

"""

import re
import os
import numpy as np
import tensorflow as tf
from glob import glob
import argparse
from collections import Counter
from artifice.utils import img, dataset, augment
from artifice.semantic_segmentation import UNet
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# breed_name -> (class, count)
breeds = {
  'Abyssinian' : (2,198),
  'Bengal' : (2,200),
  'Birman' : (2,200),
  'Bombay' : (2,200),
  'British_Shorthair' : (2,184),
  'Egyptian_Mau' : (2,200),
  'Maine_Coon' : (2,190),
  'Persian' : (2,200),
  'Ragdoll' : (2,200),
  'Russian_Blue' : (2,200),
  'Siamese' : (2,199),
  'Sphynx' : (2,200),
  'american_bulldog' : (1,200),
  'american_pit_bull_terrier' : (1,200),
  'basset_hound' : (1,200),
  'beagle' : (1,200),
  'boxer' : (1,199),
  'chihuahua' : (1,200),
  'english_cocker_spaniel' : (1,196),
  'english_setter' : (1,200),
  'german_shorthaired' : (1,200),
  'great_pyrenees' : (1,200),
  'havanese' : (1,200),
  'japanese_chin' : (1,200),
  'keeshond' : (1,199),
  'leonberger' : (1,200),
  'miniature_pinscher' : (1,200),
  'newfoundland' : (1,196),
  'pomeranian' : (1,200),
  'pug' : (1,200),
  'saint_bernard' : (1,200),
  'samoyed' : (1,200),
  'scottish_terrier' : (1,199),
  'shiba_inu' : (1,200),
  'staffordshire_bull_terrier' : (1,189),
  'wheaten_terrier' : (1,200),
  'yorkshire_terrier' : (1,200)
}


def breed_from_filename(file_name):
  """Gets the integer class label from a file name.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.((png)|(jpg)))', file_name, re.I)
  return match.groups()[0]

def class_from_filename(file_name):
  return breeds[breed_from_filename(file_name)][0]

def breed_class(breed):
  return breeds[breed][0]

def breed_prevalence(breed, scarcity=0, test_per_breed=0):
  """Calculate the prevalence of the breed, given scarcity level, and excluding
  test_per_breed examples.

  Default inputs yield the total prevalence across both test, train sets.

  """
  prevalence = breeds[breed][1] - test_per_breed
  return int((1 - scarcity) * prevalence)


def make_dataset(record, image_names, annotation_names, 
                 shape=None, augmentation=None, overwrite=False):
  """Helper function for making datasets, using absolute lists of files.

  :record: name of output tfrecord file to write.
  :image_names: list of file paths to images
  :image_names: list of file paths to annotations, in corresponding order
  :shape: image_names
  :augmentation: an augmentation function, which takes in the image and
    annotation and returns a list like [(image, annotation),].

  """

  if not overwrite and os.path.exists(record):
    raise RuntimeError(f"'{record}' already exists; set --overwrite to ignore")

  writer = tf.python_io.TFRecordWriter(record)
  N = len(image_names)
  indices = np.random.permutation(N)
  cnt = 0
  for i in indices:
    logging.info(f"Writing example {cnt} of {N}.")
    
    image = img.open_as_array(image_names[i])
    mask = img.open_as_array(annotation_names[i])
    annotation = np.zeros_like(mask, dtype=np.uint8)
    label = class_from_filename(os.path.basename(annotation_names[i]))
    annotation[mask == 1] = label # animal
    annotation[mask == 2] = 0     # background
    annotation[mask == 3] = 0     # boundary -> background
    
    if shape is not None:
      logging.debug(f"resizing images to shape {shape}")
      image = img.resize(image, (shape[0], shape[1], 3))
      annotation = img.resize(annotation, shape, label=label)

    if augmentation is not None:
      for scene in augmentation(image, annotation):
        e = dataset.example_string_from_scene(*scene)
        writer.write(e)
    else:
      e = dataset.example_string_from_scene(image, annotation)
      writer.write(e)
    cnt += 1

  writer.close()


def create_training_set(train_record_name,
                        images_path='data/pets/images',
                        annotations_path='data/pets/annotations/trimaps',
                        scarcity=0,
                        test_per_breed=5,
                        **kwargs):
  """Create a new training set with the given scarcity and augmentation.

  :scarcity: A float in [0,1). Measure of how scarce to make the dataset,
    removing a proportional number from each breed.
  :test_per_breed: used to calculate the correct scarcity
  :augmentation:
  :overwrite:
  
  """
  assert(0 <= scarcity < 1)

  image_names = np.array(sorted(glob(os.path.join(images_path, "*.jpg"))))
  annotation_names = np.array(sorted(glob(os.path.join(annotations_path, "*.png"))))

  train_indices = []
  counter = Counter()
  current_breed = ""
  current_breed_count = 0
  for i in range(len(image_names)):
    breed = breed_from_filename(os.path.basename(image_names[i]))
    counter.update([breed])
    if counter[breed] < breed_prevalence(breed, scarcity, test_per_breed):
      train_indices.append(i)

  logging.info(f"Creating '{train_record_name}'...")
  make_dataset(train_record_name, 
               image_names[train_indices], 
               annotation_names[train_indices],
               **kwargs)
  logging.info("Finished.\n")


def create_original_dataset(train_record_name='data/pets/train.tfrecord',
                            test_record_name='data/pets/test.tfrecord',
                            images_path='data/pets/images',
                            annotations_path='data/pets/annotations/trimaps',
                            test_per_breed=5,
                            **kwargs):
  """Create the dataset from base pets images. Perform test_train split. 

  :test_per_breed: number of examples to hold back from each breed for
    testing.
  :shape: first two dimensions of image_shape to enforce
  """

  image_names = np.array(sorted(glob(os.path.join(images_path, "*.jpg"))))
  annotation_names = np.array(sorted(glob(os.path.join(annotations_path, "*.png"))))

  train_indices = []
  test_indices = []
  counter = Counter()
  for i in range(len(image_names)):
    # Reserve the last 'test_per_breed' examples for testing
    # TODO: debug why this made 263 files in test set
    # Current training set has 7127 files. At least mutex.
    breed = breed_from_filename(os.path.basename(image_names[i]))
    prevalence = breed_prevalence(breed)
    counter.update([breed])
    if counter[breed] >= prevalence - test_per_breed:
      test_indices.append(i)
    else:
      train_indices.append(i)
    
  logging.info(f"Creating '{train_record_name}' with {len(train_indices)} examples...")
  make_dataset(train_record_name, 
               image_names[train_indices], 
               annotation_names[train_indices],
               **kwargs)
  logging.info("Finished.\n")

  logging.info(f"Creating '{test_record_name}' with {len(test_indices)} examples...")  
  make_dataset(test_record_name,
               image_names[test_indices],
               annotation_names[test_indices],
               **kwargs)
  logging.info("Finished.")

# Temporary stuff, pass in as command args instead
train_record_name = "data/pets/train.tfrecord"
test_record_name = "data/pets/test.tfrecord"
shape = (512, 512)
image_shape = (512, 512, 3)
num_test = 300
test_per_breed = 5

def cmd_data(args):
  # TODO: configure command line args for this
  if args.original:
    create_original_dataset(train_record_name=args.train_record,
                            test_record_name=args.test_record,
                            test_per_breed=test_per_breed,
                            shape=shape,
                            overwrite=args.overwrite)
  else:
    create_training_set(args.train_record,
                        scarcity=args.scarcity,
                        test_per_breed=test_per_breed,
                        augmentation=None,
                        overwrite=args.overwrite,
                        shape=shape)


def cmd_train(args):
  train_data = dataset.load(train_record_name)
  test_data = dataset.load(test_record_name)
  unet = UNet(image_shape, 3, model_dir=args.model_dir)
  unet.train(train_data, test_data=test_data, overwrite=args.overwrite)
  
def cmd_predict(args):
  raise NotImplementedError("cmd_predict")
  
def main():
  parser = argparse.ArgumentParser(
    description="Semantic segmentation for the pets dataset.")
  parser.add_argument('command', choices=['data', 'train', 'predict'])
  parser.add_argument('--train-record', '-o', nargs='?',
                      default='data/pets/train.tfrecord', 
                      const='data/pets/train.tfrecord',
                      help='tfrecord name for training set')
  parser.add_argument('--test-record', '-t', nargs='?',
                      default='data/pets/test.tfrecord', 
                      const='data/pets/test.tfrecord',
                      help='tfrecord name for test set')

  # Data options
  dataset_group = parser.add_mutually_exclusive_group()
  dataset_group.add_argument('--original', action='store_true',
                             help='create the original train, test datasets')
  dataset_group.add_argument('--scarcity', '-s', nargs='?',
                             default=0, const=0, type=float,
                             help='measure of dataset scarcity in [0,1)')
  
  # Training options
  train_group = parser.add_argument_group(title="train",
                                         description="model training options")
  train_group.add_argument('--model-dir', '-m', nargs='?', 
                           default='models/pets', const='models/pets',
                           help='save model checkpoints to MODEL_DIR')
  train_group.add_argument('--overwrite', '-f', action='store_true',
                           help='overwrite objects')

  # prediction options
  predict_group = parser.add_argument_group(title="predict",
                                            description="model prediction options")

  args = parser.parse_args()

  if args.command == 'data':
    cmd_data(args)
  elif args.command == 'train':
    cmd_train(args)
  elif args.command == 'predict':
    cmd_predict(args)
  else:
    raise RuntimeError()


if __name__ == "__main__":
  main()
