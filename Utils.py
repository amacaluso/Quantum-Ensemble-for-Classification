# Classical packages
import jupyter
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split

from IPython.display import Image
from IPython.core.display import HTML

import pandas as pd
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer, execute



def create_dir (path):
    if not os.path.exists(path):
        print('The directory', path, 'does not exist and will be created')
        os.makedirs(path)
    else:
        print('The directory', path, ' already exists')


def save_dict(d, name = 'dict'):
    df = pd.DataFrame(list(d.items()))
    name = name + '_' + str(np.random.randint(10**6)) + '.csv'
    df.to_csv(name)


def normalize_custom(x, C =1):
    M = x[0] ** 2 + x[1] ** 2

    x_normed = [
        1 / np.sqrt(M * C) * complex(x[0], 0),  # 00
        1 / np.sqrt(M * C) * complex(x[1], 0),  # 01
    ]
    return x_normed

def add_label( d, label = '0'):
    try:
        d[label]
        print( 'Label', label, 'exists')
    except:
        d[label] = 0
    return d

def exec_simulator(qc, n_shots = 1000):
    # QASM simulation
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots = n_shots)
    results = job.result()
    answer = results.get_counts(qc)
    return answer


def cos_classifier(train, test, label_train, printing=False):
    # x_train = train
    # x_new = test
    # y_train = label_train
    c = ClassicalRegister(1)
    x_train = QuantumRegister(1, 'x_train')
    x_test = QuantumRegister(1, 'x_test')
    y_train = QuantumRegister(1, 'y_train')
    qc = QuantumCircuit(x_train, x_test, y_train, c)
    qc.initialize(train, [x_train[0]])
    qc.initialize(test, [x_test[0]])
    qc.initialize(label_train, [y_train[0]])
    qc.barrier()
    qc.h(y_train)
    qc.cswap(y_train, x_train, x_test)
    qc.h(y_train)
    qc.barrier()
    qc.measure(y_train, c)
    if printing:
        print(qc)
    return qc

def plot_cls( dictionary, title = 'Test point classification' ):
    N = len(dictionary)
    fig, ax = plt.subplots()
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35         # the width of the bars
    prob_0 = [p['0']/(p['0'] + p['1']) for p in dictionary]
    prob_1 = [p['1']/(p['0'] + p['1']) for p in dictionary]
    label = [l['label'] for l in dictionary]
    pl1 = ax.bar(ind, prob_0, width, bottom=0)
    pl2 = ax.bar(ind + width, prob_1, width, bottom=0)
    ax.set_title( title )
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels( label )
    ax.legend((pl1[0], pl2[0]), ('P(y=0)', 'P(y=1)'))
    ax.autoscale_view()
    plt.show()


def load_data_custom(X_data=None, Y_data=None, x_test=None):
    y_class0 = [1, 0]
    y_class1 = [0, 1]

    # Training Set
    if X_data is None:
        x1 = [1, 3];
        y1 = [1, 0]
        x2 = [-2, 2];
        y2 = [0, 1]
        x3 = [3, 0];
        y3 = [1, 0]
        x4 = [3, 1];
        y4 = [0, 1]
        X_data = [x1, x2, x3, x4]
        Y_data = [y1, y2, y3, y4]

    if x_test is None:
        x_test = [2, 2]

    print(X_data)

    V = np.array([x1, x3, x2, x4, x_test])
    origin = [0], [0]  # origin point
    plt.quiver(*origin, V[:, 0], V[:, 1], color=['tan', 'tan', 'g', 'g', 'red'], scale=10)
    plt.show()

    X_data = [normalize_custom(x) for x in X_data]
    x_test = normalize_custom(x_test)

    return X_data, Y_data, x_test


def pdf(url):
    return HTML('<embed src="%s" type="application/pdf" width="100%%" height="600px" />' % url)



def predict_cos(M):
    M0 = (M['0'] / (M['0'] + M['1']))-.2
    M1 = 1 - M0
    return [M0, M1]



def retrieve_proba(r):
    try:
        p0 = r['0'] / (r['0'] + r['1'])
        p1 = 1 - p0
    except:
        if r.keys()[0] == '0':
            p0 = 1;
            p1 = 0
        elif r.keys()[0] == '1':
            p0 = 0;
            p1 = 1
    return [p0, p1]





def multivariateGrid(col_x, col_y, col_k, df, k_is_color=False, scatter_alpha=.5):
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=name
        g.plot_joint(
            colored_scatter(df_group[col_x],df_group[col_y],color),
        )
        sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            color=color,
        )
        sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            color=color,
            vertical=True
        )
    # Do also global Hist:
    sns.distplot(
        df[col_x].values,
        ax=g.ax_marg_x,
        color='grey'
    )
    sns.distplot(
        df[col_y].values.ravel(),
        ax=g.ax_marg_y,
        color='grey',
        vertical=True
    )
    plt.tight_layout()
    plt.xlabel(r'$x_1$', fontsize = 14)
    plt.ylabel(r'$x_2$', fontsize=14, rotation = 0)
    plt.legend(legends, fontsize=14, loc = 'lower left')
    plt.grid(alpha=0.3)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    plt.savefig('data/data.png', dpi = 300, bbox_inches = "tight")
    plt.show()
    plt.close()




def load_data(n=100, centers=[[0.5, .1], [.1, 0.5]],
              std=.20, seed=4552, plot=True, save=True):
    X, y = datasets.make_blobs(n_samples=n, centers=centers,
                               n_features=2, center_box=(0, 1),
                               cluster_std=std, random_state=seed)

    if plot:
        columns = ['$x_1$', '$x_2$', 'Y']
        data = pd.concat([pd.DataFrame(X), pd.DataFrame(np.where(y == 0, 'class 0', 'class 1'))], axis=1)
        data.columns = columns
        multivariateGrid('$x_1$', '$x_2$', 'Y', df=data)
    if save:
        data.to_csv('data/all_data.csv', index=False)
    return X, y

def label_to_array(y):
    Y = []
    for el in y:
        if el == 0: Y.append([1, 0])
        else: Y.append([0, 1])
    Y = np.asarray(Y)
    return Y




def evaluation_metrics(predictions, X_test, y_test, save=True):
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




### Ensemble
def ensemble_fixed_U(X_data, Y_data, x_test, d = 2 ):
    #d = 2  # number of control qubits
    n_obs = len(X_data)

    if n_obs != len(Y_data):
        return print('Error: in the input size')

    n_reg = d + 2 * n_obs + 1  # total number of registers

    control = QuantumRegister(d)
    data = QuantumRegister(n_obs, 'x')
    labels = QuantumRegister(n_obs, 'y')
    test = QuantumRegister(1, 'test')
    c = ClassicalRegister(1)

    qc = QuantumCircuit(control, data, labels, test, c)

    for index in range(n_obs):
        qc.initialize(X_data[index], [data[index]])
        qc.initialize(Y_data[index], [labels[index]])

    for i in range(d):
        qc.h(control[i])

    U1 = [0, 2]  # np.random.choice(range(4), 2, replace=False)
    U2 = [1, 3]  # np.random.choice(range(4), 2, replace=False)
    # U3 = [0, 0]  # np.random.choice(range(4), 2, replace=False)
    U4 = [2,3]  # np.random.choice(range(4), 2, replace=False)

    qc.barrier()

    # U1
    qc.cswap(control[0], data[int(U1[0])], data[int(U1[1])])
    qc.cswap(control[0], labels[int(U1[0])], labels[int(U1[1])])

    qc.x(control[0])

    # U2
    qc.cswap(control[0], data[int(U2[0])], data[int(U2[1])])
    qc.cswap(control[0], labels[int(U2[0])], labels[int(U2[1])])

    qc.barrier()

    # U3
    # qc.cswap(control[1], data[int(U3[0])], data[int(U3[1])])
    # qc.cswap(control[1], labels[int(U3[0])], labels[int(U3[1])])

    qc.x(control[1])

    # U4
    qc.cswap(control[1], data[int(U4[0])], data[int(U4[1])])
    qc.cswap(control[1], labels[int(U4[0])], labels[int(U4[1])])

    qc.barrier()
    qc.initialize(x_test, [test[0]])

    # C
    ix_cls = 3
    qc.h(labels[ix_cls])
    qc.cswap(labels[ix_cls], data[ix_cls], test[0])
    qc.h(labels[ix_cls])
    qc.measure(labels[ix_cls], c)
    return qc


def load_data_custom(X_data=None, Y_data=None, x_test=None):
    y_class0 = [1, 0]
    y_class1 = [0, 1]

    # Training Set
    if X_data is None:
        x1 = [1, 3]
        y1 = [1, 0]
        x2 = [-2, 2]
        y2 = [0, 1]
        x3 = [3, 0]
        y3 = [1, 0]
        x4 = [3, 1]
        y4 = [0, 1]
        X_data = [x1, x2, x3, x4]
        Y_data = [y1, y2, y3, y4]

    if x_test is None:
        x_test = [2, 2]

    print(X_data)

    V = np.array([x1, x3, x2, x4, x_test])
    origin = [0], [0]  # origin point
    plt.quiver(*origin, V[:, 0], V[:, 1], color=['tan', 'tan', 'g', 'g', 'red'], scale=10)
    plt.show()

    X_data = [normalize_custom(x) for x in X_data]
    x_test = normalize_custom(x_test)

    return X_data, Y_data, x_test


def training_set(X, Y, n=4):
    ix_y1 = np.random.choice(np.where(Y == 1)[0], int(n/2), replace=False)
    ix_y0 = np.random.choice(np.where(Y == 0)[0], int(n/2), replace=False)

    X_data = np.concatenate([X[ix_y1], X[ix_y0]])

    for i in range(len(X_data)):
        X_data[i] = normalize_custom(X_data[i])

    Y_vector = label_to_array(Y)
    Y_data = np.concatenate([Y_vector[ix_y1], Y_vector[ix_y0]])

    return X_data, Y_data


def ensemble(X_data, Y_data, x_test, n_swap=1, d=4, balanced=True):
    # d = 2  # number of control qubits
    # n_swap=2

    n_obs = len(X_data)
    if n_obs != len(Y_data):
        return print('Error: in the input size')

    n_reg = d + 2 * n_obs + 1  # total number of registers

    control = QuantumRegister(d)
    data = QuantumRegister(n_obs, 'x')
    labels = QuantumRegister(n_obs, 'y')
    test = QuantumRegister(1, 'test')
    c = ClassicalRegister(1)

    qc = QuantumCircuit(control, data, labels, test, c)

    for index in range(n_obs):
        qc.initialize(X_data[index], [data[index]])
        qc.initialize(Y_data[index], [labels[index]])

    for i in range(d):
        qc.h(control[i])

    if balanced:
        for i in range(d):
            for j in range(n_swap):
                U = np.random.choice(range(int(n_obs / 2)), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

                U = np.random.choice(range(int(n_obs / 2), n_obs), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

            qc.x(control[i])

            for j in range(n_swap):
                U = np.random.choice(range(int(n_obs / 2)), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

                U = np.random.choice(range(int(n_obs / 2), n_obs), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

                qc.barrier()
    else:
        for i in range(d):
            for j in range(n_swap):
                U = np.random.choice(range(n_obs), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

            qc.x(control[i])

            for j in range(n_swap):
                U = np.random.choice(range(n_obs), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

                qc.barrier()

    qc.initialize(x_test, [test[0]])

    # C
    ix_cls = n_obs - 1
    qc.h(labels[ix_cls])
    qc.cswap(labels[ix_cls], data[ix_cls], test[0])
    qc.h(labels[ix_cls])
    qc.measure(labels[ix_cls], c)
    return qc