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
from skimage.transform import resize
import argparse
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
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.(png)|(jpg))', file_name, re.I)
  return match.groups()[0]

def class_from_filename(file_name):
  return breeds[breed_from_filename(file_name)][0]

def breed_class(breed):
  return breeds[breed][0]

def breed_prevalence(breed, scarcity=0):
  prevalence = breeds[breed][1]
  return int((1 - scarcity) * prevalence)


def make_dataset(record, image_names, annotation_names, 
                 shape=None, augmentation=None):
  """Helper function for making datasets, using absolute lists of files.

  :record: name of output tfrecord file to write.
  :image_names: list of file paths to images
  :iamge_names: list of file paths to annotations, in corresponding order
  :shape: image_names
  :augmentation: an augmentation function, which takes in the image and
    annotation and returns a list like [(image, annotation),].

  """

  writer = tf.python_io.TFRecordWriter(record)
  N = len(image_names)
  for i in range(N):
    logging.info(f"Writing example {i} of {N}.")
    
    image = img.open_as_array(image_names[i])
    mask = img.open_as_array(annotation_names[i])
    annotation = np.zeros_like(mask, dtype=np.uint8)
    label = class_from_filename(os.path.basename(annotation_names[i]))
    annotation[mask == 1] = label # animal
    annotation[mask == 2] = 0     # background
    annotation[mask == 3] = 0     # boundary -> background
    
    if shape is not None:
      logging.debug(f"resizing images to shape {shape}")
      image = (255 * resize(image, (shape[0], shape[1], 3), mode='reflect'))
      annotation = (255*resize(annotation, shape, mode='reflect'))
      image = image.astype(np.uint8)
      annotation = annotation.astype(np.uint8)
      # Clean up bad classes from interpolation.
      bad_label = 1 if label == 2 else 2
      annotation[annotation == bad_label] = label

    if augmentation is not None:
      for scene in augmentation(image, annotation):
        e = dataset.example_string_from_scene(*scene)
        writer.write(e)
    else:
      e = dataset.example_string_from_scene(image, annotation)
      writer.write(e)

  writer.close()


def create_training_set(train_record_name,
                        images_path='data/pets/images',
                        annotations_path='data/pets/annotations/trimaps',
                        split_file='data/pets/test_train_split.npy',
                        scarcity=0,
                        augmentation=None):
  """Create a new training set.

  :scarcity: A float in [0,1). Measure of how scarce to make the dataset,
    removing a proportional number from each breed.
  :augmentation:
  
  """
  if not os.path.exists(split_file):
    raise RuntimeError(f"missing '{split_file}':create original dataset first")

  pass


  

def create_original_dataset(train_record_name='data/pets/train.tfrecord',
                            test_record_name='data/pets/test.tfrecord',
                            images_path='data/pets/images',
                            annotations_path='data/pets/annotations/trimaps',
                            split_file='data/pets/test_train_split.npy',
                            overwrite=False,
                            num_test=300,
                            test_per_breed=5,
                            shape=(512,512)):
  """Create the dataset from base pets images. Perform test_train split. 

  :split_file: npy file with indices to use for training set.
  :overwrite: overwrite indices files if they exist. Ignored if either file does
    not exist.
  :test_per_breed: number of examples to hold back from each breed for
    testing.
  :shape: first two dimensions of image_shape to enforce

  TODO: keep a deterministic test-train split.

  """

  image_names = np.array(sorted(glob(os.path.join(images_path, "*.jpg"))))
  annotation_names = np.array(sorted(glob(os.path.join(annotations_path, "*.png"))))
  N = len(image_names)
  
  if overwrite or not os.path.exists(split_file):
    logging.info(f"Creating test-train split with {num_test} test images.")
    perm = np.random.permutation(N)
    train_indices = perm[num_test:]
    test_indices = perm[:num_test]
    np.save(split_file, (train_indices, test_indices))
  else:
    logging.info("Loading saved test-train split.")
    train_indices, test_indices = np.load(split_file)
  
  logging.info(f"Creating '{train_record_name}' with {N - num_test} examples...")
  make_dataset(train_record_name, 
               image_names[train_indices], 
               annotation_names[train_indices])
  logging.info("Finished.\n")

  logging.info(f"Creating '{test_record_name}' with {num_test} examples...")  
  make_dataset(test_record_name,
               image_names[test_indices],
               annotation_names[test_indices])
  logging.info("Finished.")


# Temporary stuff, pass in as command args instead
train_record_name = "data/pets/train.tfrecord"
test_record_name = "data/pets/test.tfrecord"
image_shape = (512, 512, 3)
split_file = "data/pets/test_train_split.npy"
num_test = 300

def cmd_data(args):
  # TODO: configure command line args for this
  create_original_dataset(train_record_name=train_record_name,
                          test_record_name=test_record_name,
                          split_file=split_file,
                          num_test=num_test,
                          overwrite=args.overwrite,
                          shape=(image_shape[0], image_shape[1]))

def cmd_train(args):
  train_data = dataset.load(train_record_name)
  test_data = dataset.load(test_record_name)
  unet = UNet(image_shape, 3, model_dir=args.model_dir)
  unet.train(train_data, test_data=test_data, overwrite=args.overwrite)
  
def cmd_test(args):
  pass
  
def main():
  parser = argparse.ArgumentParser(
    description="Semantic segmentation for the pets dataset.")
  parser.add_argument('command', choices=['data', 'train', 'test'])
  parser.add_argument('--model-dir', '-m', nargs='?', 
                      default='models/pets', const='models/pets',
                      help='save model checkpoints to MODEL_DIR')
  parser.add_argument('--overwrite', action='store_true',
                      help='data: reselect test set\
train: overwrite MODEL_DIR, if it exists')

  args = parser.parse_args()

  if args.command == 'data':
    cmd_data(args)
  elif args.command == 'train':
    cmd_train(args)
  else:
    raise RuntimeError()


if __name__ == "__main__":
  main()
