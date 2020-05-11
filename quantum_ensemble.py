import sys

from Utils import *
from modeling import *

sys.path.insert(1, '../')

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
provider.backends()
backend = provider.get_backend('ibmq_qasm_simulator')

# create_dir('data')
# create_dir('output')
# create_dir('IMG')

n_shots = 500
n_swap = 1

# n_train = 2
# d = 1
# std = .3

balanced = True

# seed = 565
np.random.seed(seed)

n = 200
test_size = .1

X, y = load_data(n=n, std=std, save=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

Y_vector_train = label_to_array(y_train)
Y_vector_test = label_to_array(y_test)

print("Size Training Set: ", len(X_train))
print("Size Test Set: ", len(X_test))

accuracy = []
predictions = []

for x_test, y_ts in zip(X_test, Y_vector_test):
    X_data, Y_data = training_set(X_train, y_train, n=n_train)
    x_test = normalize_custom(x_test)

    qc = ensemble(X_data, Y_data, x_test, n_swap=n_swap, d=d, balanced=balanced)
    qc = transpile(qc, basis_gates = ['u1', 'u2', 'u3', 'cx'], optimization_level=3)
    # r = exec_simulator(qc, n_shots=n_shots)

    job = execute(qc, backend, shots=n_shots)
    results = job.result()
    r = results.get_counts(qc)

    predictions.append(retrieve_proba(r))
    # print(retrieve_proba(r), y_ts)

a, b = evaluation_metrics(predictions, X_test, y_test)
print(a,b)

file = open("output/results_ensemble.csv", 'a')
file.write("%d, %d, %d, %d, %s,%f, %f, %f, %f, %d\n" % (n, n_train, n_swap, d, balanced, test_size, std, a, b, seed))
file.close()
