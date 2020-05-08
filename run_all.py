from past.builtins import execfile

# 565: std = .25, .30 and all d=1
# 36: solo d=2 std .1

seeds = [565, 36, 1252, 7]
d_vector = [2, 3, 4, 1]
stds = [.1, .15, .20, .25, .30]


for seed in seeds:
    for d in d_vector:
        for std in stds:
            n_train = 2**d
            if n_train > 8:
                n_train = 8
            print(seed, d, n_train, std)
            execfile('quantum_ensemble.py')
