"""Make all the scripts for training and analysis.

Unlike most artifice scripts, this should be run from ROOT/batch."""

import os
import itertools

train_template = """#!/bin/bash

#SBATCH --job-name={mode}
#SBATCH --output={out_name}
#SBATCH --error={err_name}
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=pi-glk
#SBATCH --mem-per-cpu=32000


module unload python
module load ffmpeg
module load cuda/9.0
module load Anaconda3/2018.12
source activate tf-gpu-1.12.0
module load povray/3.7

cd /project2/glk/artifice

epochs={epochs}
mode={mode}
data={data}
subset_size={subset_size} # ignored if mode is 'full'
query_size={query_size}   # ignored if mode is 'full'

echo "Starting training..."
python artifice.py train --mode $mode -i data/$data \\
       --overwrite -e $epochs \\
       -m models/${{data}}_${{mode}}{subset_addon} \\
       --subset-size $subset_size --query-size $query_size \\
       --verbose 2 --keras-verbose 2
echo "Finished."

"""

analysis_template = """#!/bin/bash

#SBATCH --job-name={mode}
#SBATCH --output={out_name}
#SBATCH --error={err_name}
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=pi-glk
#SBATCH --mem-per-cpu=32000


module unload python
module load ffmpeg
module load cuda/9.0
module load Anaconda3/2018.12
source activate tf-gpu-1.12.0
module load povray/3.7

cd /project2/glk/artifice

mode={mode}
data={data}
subset_size={subset_size}

# echo "Starting coupled_spheres..."
# python artifice.py {cmd} --mode $mode -i data/coupled_spheres \\
#        -m models/${{data}}_${{mode}}{subset_addon} \\
#        --verbose 2 --keras-verbose 2
# echo "Finished."

# echo "Starting coupled_spheres_tethered..."
# python artifice.py {cmd} --mode $mode -i data/coupled_spheres_tethered \\
#        -m models/${{data}}_${{mode}}{subset_addon} \\
#        --verbose 2 --keras-verbose 2
# echo "Finished."

echo "Starting waltzing_spheres..."
python artifice.py {cmd} --mode $mode -i data/waltzing_spheres \\
       -m models/${{data}}_${{mode}}{subset_addon} \\
       --splits 0 0 2401 \\
       --verbose 2 --keras-verbose 2
echo "Finished."
"""


num_active_epochs = 10

which_epochs = [20]
modes = ['full', 'random', 'active',
         'augmented-full', 'augmented-random', 'augmented-active']
datas = ['coupled_spheres', 'coupled_spheres_tethered']
subset_sizes = [10, 100]

for t in itertools.product(which_epochs, modes, datas, subset_sizes):
  print(t)
  epochs, mode, data, subset_size = t
  dir_name = f"train/{mode}_{data}" + ("" if 'full' in mode
                                       else f"_subset{subset_size}")
  if not os.path.exists(dir_name):
    os.mkdir(dir_name)

  subset_addon = "" if 'full' in mode else "_subset${subset_size}"
  query_size = subset_size // num_active_epochs
  out_name = os.path.join(os.getcwd(), dir_name, 'train.out')
  err_name = os.path.join(os.getcwd(), dir_name, 'train.err')
  script = train_template.format(
    epochs=epochs, mode=mode, data=data, subset_size=subset_size,
    query_size=query_size, subset_addon=subset_addon,
    out_name=out_name, err_name=err_name)
  with open(os.path.join(dir_name, 'train.batch'), 'w') as f:
    f.write(script)

  cmd = 'detect'
  out_name = os.path.join(os.getcwd(), dir_name, f'{cmd}.out')
  err_name = os.path.join(os.getcwd(), dir_name, f'{cmd}.err')  
  script = analysis_template.format(
    cmd=cmd, mode=mode, data=data, subset_size=subset_size,
    subset_addon=subset_addon, out_name=out_name, err_name=err_name)
  with open(os.path.join(dir_name, 'detect.batch'), 'w') as f:
    f.write(script)

  cmd = 'visualize'
  out_name = os.path.join(os.getcwd(), dir_name, f'{cmd}.out')
  err_name = os.path.join(os.getcwd(), dir_name, f'{cmd}.err')  
  script = analysis_template.format(
    cmd=cmd, mode=mode, data=data, subset_size=subset_size,
    subset_addon=subset_addon, out_name=out_name, err_name=err_name)
  with open(os.path.join(dir_name, 'visualize.batch'), 'w') as f:
    f.write(script)
