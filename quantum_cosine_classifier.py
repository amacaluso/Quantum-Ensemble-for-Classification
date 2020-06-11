import sys

sys.path.insert(1, '../')

from Utils import *
from modeling import *

d=0
n_train=1
#seed=962
#std=.3
np.random.seed(seed)

# create_dir('data')
# create_dir('output')

n_shots = 1000
n_swap = 1
balanced = True

n = 200
test_size = .1

X, y = load_data(n=n, std=std)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

Y_vector_train = label_to_array(y_train)
Y_vector_test = label_to_array(y_test)

print("Size Training Set: ", len(X_train))
print("Size Test Set: ", len(X_test))

predictions = []

for x_test, y_ts in zip(X_test, Y_vector_test):
    ix = np.random.choice(int(n*(1-test_size)), 1)[0]
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
print(a,b)

file = open("output/old/results_ensemble_fixed_seed_all.csv", 'a')
file.write("%d, %d, %d, %d, %s,%f, %f, %f, %f, %d\n" % (n, n_train, n_swap, d, balanced, test_size, std, a, b, seed))
file.close()

