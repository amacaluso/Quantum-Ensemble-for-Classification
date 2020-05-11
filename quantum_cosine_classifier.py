import sys

from Utils import *
from modeling import *

sys.path.insert(1, '../')

n_shots = 500
n_swap = 1

# n_train = 2
# d = 1
# seed = 565
# std = .1

np.random.seed(seed)

balanced = True
n = 200
test_size = .1
n_train=1

X, y = load_data(n=n, std=std)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

Y_vector_train = label_to_array(y_train)
Y_vector_test = label_to_array(y_test)

print("Size Training Set: ", len(X_train))
print("Size Test Set: ", len(X_test))

accuracy = []

# for i in range(100):
#     n = range(len(X_train))
#     TP = 0
predictions = []
probabilities = []

for x_test, y_ts in zip(X_test, Y_vector_test):
    ix = np.random.choice(n, 1)[0]
    x_train = X_train[ix]
    x_tr = normalize_custom(x_train)
    y_tr = Y_vector_train[ix]
    x_ts = normalize_custom(x_test)

    qc = cos_classifier(x_tr, x_ts, y_tr)
    r = exec_simulator(qc, n_shots=n_shots)

    if '0' not in r.keys():
        r['0'] = 0
    elif '1' not in r.keys():
        r['1'] = 0

    predictions.append(retrieve_proba(r))

a, b = evaluation_metrics(predictions, X_test, y_test)


file = open("output/results_ensemble.csv", 'a')
file.write("%d, %d, %d, %d, %s,%f, %f, %f, %f, %d\n" % (n, n_train, n_swap, d, balanced, test_size, std, a, b, seed))
file.close()

