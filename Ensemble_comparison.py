import sys

from Utils import *

sys.path.insert(1, '../')

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
provider.backends()
backend = provider.get_backend('ibmq_qasm_simulator')


create_dir('data')
create_dir('output')
create_dir('IMG')

seed = 789#456 #4552
np.random.seed(seed)

X, y = load_data(n=50, std=.10)

test_size = .2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

Y_vector_train = label_to_array(y_train)
Y_vector_test = label_to_array(y_test)

print("Size Training Set: ", len(X_train))
print("Size Test Set: ", len(X_test))

accuracy = []
n_shots = 8192
predictions = []

for x_test, y_ts in zip(X_test, Y_vector_test):

    X_data, Y_data = training_set(X_train, y_train, n=8)
    x_test = normalize_custom(x_test)

    qc = ensemble(X_data, Y_data, x_test, n_swap=1, d=3, balanced=True)
    # r = exec_simulator(qc, n_shots=n_shots)

    job = execute(qc, backend, shots = n_shots)
    results = job.result()
    r = results.get_counts(qc)

    predictions.append(retrieve_proba(r))
    print(retrieve_proba(r), y_ts)

a, b = evaluation_metrics(predictions, X_test, y_test)
