from past.builtins import execfile
from Utils import *

create_dir('data')
create_dir('output')

# import random
# seeds = random.sample(range(1, 10**4), 10)
# print(seeds)

seeds = [4583, 7392, 1425, 3255, 82, 892, 3535, 5197, 5479, 6803]

# run experiment for the quantum cosine classifier with std=.3

for seed in seeds:
    execfile('quantum_cosine_classifier.py')

data = pd.read_csv('output/result_single_classifier.csv',
                   names=['n', 'n_train', 'n_swap', 'd', 'balanced',
                          'test_size', 'std', 'a', 'b', 'seed'])

np.mean(data.a)
np.std(data.a)
np.mean(data.b)
np.std(data.b)

# seeds = [4583, 7392, 1425, 3255, 82, 892, 3535, 5197, 5479, 6803]
stds = [.3,.4,.5,.6]
d_vector = [1, 2, 3, 4]

for seed in seeds:
    for d in d_vector:
        for std in stds:
            n_train = 2 ** d
            if n_train > 8:
                n_train = 8
            # print(seed, d, n_train, std)
            balanced = False
            execfile('quantum_ensemble.py')
            balanced = True
            execfile('quantum_ensemble.py')

