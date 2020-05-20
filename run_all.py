from past.builtins import execfile

# import random
# seeds = random.sample(range(1, 10**3), 5)
# print(seeds)
# 962, 274, 2, 238, 589, 469, 497, 725, 894, 988, 733, 219, 164, 315, 161


seeds = [ 161]
stds = [.4, .5]
d_vector = [2]

for seed in seeds:
    for d in d_vector:
        for std in stds:
            n_train = 2**d
            if n_train > 8:
                n_train = 8
            print(seed, d, n_train, std)
            execfile('quantum_ensemble.py')

