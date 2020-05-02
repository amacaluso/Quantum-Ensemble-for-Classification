from past.builtins import execfile

# seeds = [321, 221, 5]
# vector_d = [1, 2]
# stds = [ .15, .20, .25, .30]
#
# for seed in seeds:
#         for std in stds:
#             for d in vector_d:
#                 n_train = 2**d
#                 print(seed, std, d, n_train)
#                 execfile('quantum_ensemble.py')
#
#
# seeds = [565, 896, 321, 221, 5]
# vector_d = [3, 4, 5]
# stds = [.1, .15, .20, .25, .30]
#
# for d in vector_d:
#     for seed in seeds:
#             for std in stds:
#                     n_train = 8
#                     print(seed, std, d, n_train)
#                     execfile('quantum_ensemble.py')


seeds = [565]
vector_d = [4, 5]
stds = [.15, .20, .25, .30]

for d in vector_d:
    for seed in seeds:
            for std in stds:
                    n_train = 8
                    print(seed, std, d, n_train)
                    execfile('quantum_ensemble.py')