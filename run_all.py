from past.builtins import execfile

import random
seeds = random.sample(range(1, 10**4), 10)
print(seeds)

seeds = [5229, 2131, 6592, 9911, 2275, 2278, 6135, 7754, 301, 1242]

for seed in seeds:
    execfile('quantum_cosine_classifier.py')

from Utils import *
data = pd.read_csv('output/result_single_classifier.csv',
                   names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l'])

np.mean(data.h)
np.std(data.h)

np.mean(data.i)
np.std(data.i)

# seeds = [4583, 7392, 1425, 3255, 82, 892, 3535, 5197, 5479, 6803]
# stds = [.3,.4,.5,.6]
# d_vector = [1, 2, 3, 4]
#
# for seed in seeds:
#     for d in d_vector:
#         for std in stds:
#             n_train = 2 ** d
#             if n_train > 8:
#                 n_train = 8
#             print(seed, d, n_train, std)
#             execfile('quantum_ensemble.py')

