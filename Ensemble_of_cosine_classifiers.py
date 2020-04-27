from Utils import *
import sys
sys.path.insert(1, '../')

create_dir('data')
create_dir('output')
create_dir('IMG')

seed = 4552
np.random.seed(seed)


X, y = load_data(n=200)

test_size = .1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

Y_vector_train = label_to_array(y_train)
Y_vector_test = label_to_array(y_test)

print("Size Training Set: ", len(X_train))
print("Size Test Set: ", len(X_test))

accuracy = []
n_shots = 1000
for i in range(10):
    #initialisation
    n = range(len(X_train))
    TP = 0
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

        p0 = (r['0'] / (r['0'] + r['1']))
        p0 = p0
        p1 = 1 - p0

        predictions.append(r)
        probabilities.append(predict_cos(r))
        probs = [p0, p1]


        if predict_cos(r)[0] > predict_cos(r)[1]:
            pred = [1, 0]
            pred = np.asarray(pred)
        else:
            pred = [0, 1]
            pred = np.asarray(pred)

        if np.array_equal(pred, y_ts):
            TP = TP + 1

    accuracy.append(TP / len(X_test))

print('AVG Accuracy multiple cosine classifier:', np.mean(accuracy))
print('AVG Accuracy multiple cosine classifier:', np.std(accuracy))


#

accuracy = []
n_shots = 1000


predictions = []
for x_test, y_ts in zip(X_test, Y_vector_test):
    ix_y1 = np.random.choice(np.where(y_train == 1)[0], 2, replace=False)
    ix_y0 = np.random.choice(np.where(y_train == 0)[0], 2, replace=False)

    X_data = np.concatenate([X_train[ix_y1], X_train[ix_y0]])

    for i in range(len(X_data)):
        X_data[i] = normalize_custom(X_data[i])

    x_test = normalize_custom(x_test)
    Y_data = np.concatenate([Y_vector_train[ix_y1], Y_vector_train[ix_y0]])

    qc = ensemble_cos(X_data, Y_data, x_test)
    r = exec_simulator(qc, n_shots=1000)

    predictions.append(retrieve_proba(r))
    print(retrieve_proba(r), y_ts)



def evaluation_metrics(predictions, y_test, save=True):
    from sklearn.metrics import brier_score_loss, accuracy_score
    labels = label_to_array(y_test)


    predicted_class = np.round(np.asarray(predictions))
    acc = accuracy_score(np.array(predicted_class)[:, 1],
                         np.array(labels)[:, 1])

    columns = ['X1', 'X2', 'class0', 'class1']
    test_data = pd.concat([pd.DataFrame(X_test), pd.DataFrame(labels)], axis=1)
    p0 = [p[0] for p in predictions]
    p1 = [p[1] for p in predictions]

    test_data['p0'] = p0
    test_data['p1'] = p1
    test_data['predicted_class'] = [pred[1] for pred in predicted_class]

    test_data.columns = columns + ['p0', 'p1', 'predicted_class']

    if save:
        test_data.to_csv('output/test_data.csv', index=False)
    brier = brier_score_loss(y_test, p1)

    print('Accuracy=', acc)
    print('Brier score=', brier)
    return acc, brier





evaluation_metrics(predictions, y_test)


