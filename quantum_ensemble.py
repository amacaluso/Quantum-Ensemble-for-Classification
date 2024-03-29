import sys

sys.path.insert(1, '../')

from modeling import *

# IBMQ.load_account()
# provider = IBMQ.get_provider(hub='ibm-q')
# provider.backends()
# backend = provider.get_backend('ibmq_qasm_simulator')
backend = BasicAer.get_backend('qasm_simulator')
# d=3
# n_train=4
# seed=962
# std=.3
np.random.seed(seed)

# create_dir('data')
# create_dir('output')

n_shots = 8192
n_swap = 1

n = 200
test_size = .1

X, y = load_data(n=n, std=std)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=test_size)

Y_vector_train = label_to_array(y_train)
Y_vector_test = label_to_array(y_test)

accuracy = []
predictions = []

for x_test, y_ts in zip(X_test, Y_vector_test):
    X_data, Y_data = training_set(X_train, y_train, n=n_train, seed=seed)
    x_test = normalize_custom(x_test)

    qc = ensemble(X_data, Y_data, x_test, n_swap=n_swap, d=d, balanced=balanced)
    qc = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=1)

    job = execute(qc, backend, shots=n_shots)
    results = job.result()
    r = results.get_counts(qc)

    predictions.append(retrieve_proba(r))
    # print(retrieve_proba(r), y_ts)

a, b = evaluation_metrics(predictions, X_test, y_test, save=False)
print('seed:', seed, '   std:', std, '   d:', d, '   Accuracy:', a, '   Brier score:', b)

# file = open("output/results_ensemble.csv", 'a')
file = open("output/results_ensemble.csv", 'a')
file.write("%d, %d, %d, %d, %s,%f, %f, %f, %f, %d\n" % (n, n_train, n_swap, d, balanced, test_size, std, a, b, seed))
file.close()
