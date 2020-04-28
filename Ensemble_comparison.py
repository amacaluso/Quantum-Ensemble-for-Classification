import sys

from Utils import *

sys.path.insert(1, '../')

create_dir('data')
create_dir('output')
create_dir('IMG')

seed = 4552
np.random.seed(seed)

X, y = load_data(n=100, std=.15)

test_size = .1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

Y_vector_train = label_to_array(y_train)
Y_vector_test = label_to_array(y_test)

print("Size Training Set: ", len(X_train))
print("Size Test Set: ", len(X_test))

accuracy = []
n_shots = 20
predictions = []

for x_test, y_ts in zip(X_test, Y_vector_test):

    X_data, Y_data = training_set(X_train, y_train, n=4)
    x_test = normalize_custom(x_test)

    qc = ensemble(X_data, Y_data, x_test, n_swap=1, d=2, balanced=True)
    r = exec_simulator(qc, n_shots=n_shots)

    predictions.append(retrieve_proba(r))
    print(retrieve_proba(r), y_ts)

print(qc)
a, b = evaluation_metrics(predictions, X_test, y_test)

