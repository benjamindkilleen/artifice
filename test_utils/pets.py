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
from artifice.utils import img, dataset
from artifice.semantic_segmentation import UNet
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Class names to class mappings (capital letters are cats)
breeds = {
  'Abyssinian' : 2,
  'Bengal' : 2,
  'Birman' : 2,
  'Bombay' : 2,
  'British_Shorthair' : 2,
  'Egyptian_Mau' : 2,
  'Maine_Coon' : 2,
  'Persian' : 2,
  'Ragdoll' : 2,
  'Russian_Blue' : 2,
  'Siamese' : 2,
  'Sphynx' : 2,
  'american_bulldog' : 1,
  'american_pit_bull_terrier' : 1,
  'basset_hound' : 1,
  'beagle' : 1,
  'boxer' : 1,
  'chihuahua' : 1,
  'english_cocker_spaniel' : 1,
  'english_setter' : 1,
  'german_shorthaired' : 1,
  'great_pyrenees' : 1,
  'havanese' : 1,
  'japanese_chin' : 1,
  'keeshond' : 1,
  'leonberger' : 1,
  'miniature_pinscher' : 1,
  'newfoundland' : 1,
  'pomeranian' : 1,
  'pug' : 1,
  'saint_bernard' : 1,
  'samoyed' : 1,
  'scottish_terrier' : 1,
  'shiba_inu' : 1,
  'staffordshire_bull_terrier' : 1,
  'wheaten_terrier' : 1,
  'yorkshire_terrier' : 1
}


def class_from_filename(file_name):
  """Gets the integer class label from a file name.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.(png)|(jpg))', file_name, re.I)
  return breeds[match.groups()[0]]


def create_original_dataset(record_dir,
                            train_record='train.tfrecord',
                            test_record='test.tfrecord',
                            images_path='data/pets/images',
                            annotations_path='data/pets/annotations/trimaps',
                            split_file='data/pets/test_train_split.npy',
                            overwrite=False,
                            num_test=300,
                            shape=(512,512)):
  
  """Create the dataset from base pets images. Perform test_train split. 

  :train_indices_name: npy file with indices to use for training set
  :train_indices_name: npy file with indices to use for test set
  :overwrite: overwrite indices files if they exist. Ignored if either file does
    not exist.
  :num_test: number of examples to hold back for testing. Ignored if loading
    split from files.
  :shape: first two dimensions of image_shape to enforce

  """
  # TODO: determine if we need to shuffle the original dataset. This shouldn't
  # be necessary, of course, because we're still getting out of memory errors,
  # even with it.

  image_names = sorted(glob(os.path.join(images_path, "*.jpg")))
  annotation_names = sorted(glob(os.path.join(annotations_path, "*.png")))
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

  def make_dataset(indices, record):
    writer = tf.python_io.TFRecordWriter(record)
    cnt = 0
    for i in indices:
      logging.info(f"Writing example {cnt} of {len(indices)}.")

      logging.debug(f"opening '{image_names[i]}'")
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
      
      logging.debug(f"image shape {image.shape}, annotation shape {annotation.shape}")

      e = dataset.example_string_from_scene(image, annotation)
      writer.write(e)
      cnt += 1
  
    writer.close()
  
  train_record_name = os.path.join(record_dir, train_record)
  logging.info(f"Creating '{train_record_name}' with {N - num_test} examples...")
  make_dataset(train_indices, train_record_name)
  logging.info("Finished.\n")

  test_record_name = os.path.join(record_dir, test_record)
  logging.info(f"Creating '{test_record_name}' with {num_test} examples...")  
  make_dataset(test_indices, os.path.join(record_dir, test_record))
  logging.info("Finished.")


# Temporary stuff, pass in as command args instead
record_dir = "data/pets/"
train_record = "train.tfrecord"
test_record = "test.tfrecord"
image_shape = (512, 512, 3)
split_file = "data/pets/test_train_split.npy"
num_test = 300

def cmd_data(args):
  # TODO: configure command line args for this
  create_original_dataset(record_dir, 
                          split_file=split_file,
                          num_test=num_test,
                          overwrite=args.overwrite,
                          shape=(image_shape[0], image_shape[1]))

def cmd_train(args):
  train_data = dataset.load(os.path.join(record_dir, train_record))
  test_data = dataset.load(os.path.join(record_dir, test_record))
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
