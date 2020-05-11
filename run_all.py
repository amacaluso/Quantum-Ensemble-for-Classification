from past.builtins import execfile

# import random
# seeds = random.sample(range(1, 10**3), 5)
# print(seeds)

seeds = [274, 2, 962, 238, 589]
stds = [ .2, .3, .4]
d_vector = [ 1, 2, 3, 4]


for seed in seeds:
    for d in d_vector:
        for std in stds:
            n_train = 2**d
            if n_train > 8:
                n_train = 8
            print(seed, d, n_train, std)
            execfile('quantum_ensemble.py')


seeds = [274, 2, 962, 238, 589]
stds = [ .2, .3, .4]
