from past.builtins import execfile

import random
seeds = random.sample(range(1, 10**4), 10)
print(seeds)

seeds = [4583, 7392, 1425, 3255, 82, 892, 3535, 5197, 5479, 6803]
stds = [.3,.4,.5,.6]
d_vector = [1, 2, 3, 4]

for seed in seeds:
    for d in d_vector:
        for std in stds:
            n_train = 2 ** d
            if n_train > 8:
                n_train = 8
            print(seed, d, n_train, std)
            execfile('quantum_ensemble.py')


# seeds = [460, 208, 327, 125, 562, 979, 213, 591, 982, 587]
# seeds = [962, 274, 2, 238, 589, 469, 497, 725, 894, 988]
# seeds = [733, 219, 164, 315, 161, 68705, 97442, 96367, 28810, 68996]