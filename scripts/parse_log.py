import re
import numpy as np
import matplotlib.pyplot as plt

logfiles = [(f"/Users/Benjamin/projects/artifice/batch/train/"
             f"augmented-active_harper_spheres_tethered_subset100/train.err", 10),
            (f"/Users/Benjamin/projects/artifice/batch/train/"
             f"augmented-active_harper_spheres_tethered_subset10/train.err", 1)]

for filename, query_size in logfiles:
  with open(filename, 'r') as f:
    log = f.read()
  queries = re.findall(r'uncertainties (?P<q>.*)\n', log)
  uncertainties = -np.array([[t[1] for t in eval(q)] for q in queries])
  plt.plot(np.arange(1,10), uncertainties.mean(axis=1),
           label=f"Query size {query_size}")

plt.title("Actively Selected Queries")
plt.ylabel("Mean Peak Value for Query")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("/Users/Benjamin/projects/artifice/docs/peak_values.pdf",
            transparent=True, pad_inches=0)

