from past.builtins import execfile

# import random
# seeds = random.sample(range(1, 10**3), 5)
# print(seeds)

#fino a 238 3 8 0.2

seeds = [ 296, 77, 810, 253]
d_vector = [1, 2, 3, 4]
stds = [.1, .15, .20, .25, .30]


for seed in seeds:
    for d in d_vector:
        for std in stds:
            n_train = 2**d
            if n_train > 8:
                n_train = 8
            print(seed, d, n_train, std)
            execfile('quantum_ensemble.py')
