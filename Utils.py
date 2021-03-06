# Classical packages
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import random
from numpy.random import uniform

from scipy.stats import ttest_ind
from IPython.display import Image


import warnings
warnings.filterwarnings('ignore')

def create_dir(path):
    if not os.path.exists(path):
        print('The directory', path, 'does not exist and will be created')
        os.makedirs(path)
    else:
        print('The directory', path, ' already exists')


def save_dict(d, name='dict'):
    df = pd.DataFrame(list(d.items()))
    name = name + '_' + str(np.random.randint(10 ** 6)) + '.csv'
    df.to_csv(name)


def normalize_custom(x, C=1):
    M = x[0] ** 2 + x[1] ** 2

    x_normed = [
        1 / np.sqrt(M * C) * complex(x[0], 0),  # 00
        1 / np.sqrt(M * C) * complex(x[1], 0),  # 01
    ]
    return x_normed


def add_label(d, label='0'):
    try:
        d[label]
        print('Label', label, 'exists')
    except:
        d[label] = 0
    return d


# def plot_cls(predictions,
#              labels=['$f_1$', '$f_2$', '$f_3$', '$f_4$', 'AVG', '$Ensemble$'],
#              title='Test point classification',
#              file='ens_vs_single.png'):
#     N = len(predictions)
#     fig, ax = plt.subplots()
#     ind = np.arange(N)  # the x locations for the groups
#     width = 0.35  # the width of the bars
#     prob_0 = [p[0] for p in predictions]
#     prob_1 = [p[1] for p in predictions]
#     # label = [l['label'] for l in dictionary]
#     pl1 = ax.bar(ind, prob_0, width, bottom=0)
#     pl2 = ax.bar(ind + width, prob_1, width, bottom=0)
#     ax.set_title(title)
#     ax.set_xticks(ind + width / 2)
#     ax.set_xticklabels(labels, size=10)
#     ax.legend((pl1[0], pl2[0]), ('P(y=0)', 'P(y=1)'))
#     ax.autoscale_view()
#     plt.ylim(0, 1)
#     plt.grid(alpha=.2)
#     plt.savefig('output/' + file + '.png', dpi=200)
#     plt.show()

def plot_cls(predictions,
             #labels=['$f_1$', '$f_2$', '$f_3$', '$f_4$', 'AVG', '$Ensemble$'],
             title='Test point classification',
             file=None):
    N = len(predictions)
    fig, ax = plt.subplots()
    plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars
    prob_0 = [p[0] for p in predictions]
    prob_1 = [p[1] for p in predictions]
    # label = [l['label'] for l in dictionary]
    pl1 = ax.bar(ind, prob_0, width, bottom=0)
    pl2 = ax.bar(ind + width, prob_1, width, bottom=0)
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels([r'$f_1$', r'$f_2$', r'$f_3$', r'$f_4$', 'AVG', 'Ensemble'], size=15)
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],size=15)
    ax.legend((pl1[0], pl2[0]), (r'$P(\tilde{y}=0)$', r'$P(\tilde{y}=1)$'), prop=dict(size=14))
    ax.autoscale_view()
    plt.ylim(0, 1)
    plt.xlabel('Classifier')
    #plt.xlabel(r'$P(\tilde{y})$')
    plt.grid(alpha=.2)
    ax.tick_params(pad=5)
    if file is not None:
        plt.savefig('output/' + file + '.png', dpi=200)
    plt.show()



def load_data_custom(X_data=None, Y_data=None, x_test=None, normalize=True):
    # Training Set
    if X_data is None:
        x1 = [1, 3]
        x2 = [-2, 2]
        x3 = [3, 0]
        x4 = [3, 1]
        X_data = [x1, x2, x3, x4]

    if Y_data is None:
        y1 = [1, 0]
        y2 = [0, 1]
        y3 = [1, 0]
        y4 = [0, 1]
        Y_data = [y1, y2, y3, y4]

    if x_test is None:
        x_test = [2, 2]

    # print(X_data)

    # V = np.array([x1, x3, x2, x4, x_test])
    # origin = [0], [0]  # origin point
    # plt.quiver(*origin, V[:, 0], V[:, 1], color=['tan', 'tan', 'g', 'g', 'red'], scale=10)
    # plt.show()

    if normalize:
        X_data = [normalize_custom(x) for x in X_data]
        x_test = normalize_custom(x_test)

    return X_data, Y_data, x_test


def pdf(url):
    return HTML('<embed src="%s" type="application/pdf" width="100%%" height="600px" />' % url)


def predict_cos(M):
    M0 = (M['0'] / (M['0'] + M['1'])) - .2
    M1 = 1 - M0
    return [M0, M1]


def retrieve_proba(r):
    try:
        p0 = r['0'] / (r['0'] + r['1'])
        p1 = 1 - p0
    except:
        if list(r.keys())[0] == '0':
            p0 = 1
            p1 = 0
        elif list(r.keys())[0] == '1':
            p0 = 0
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
    legends = []
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color = name
        g.plot_joint(
            colored_scatter(df_group[col_x], df_group[col_y], color),
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
    plt.xlabel(r'$x_1$', fontsize=14)
    plt.ylabel(r'$x_2$', fontsize=14, rotation=0)
    plt.legend(legends, fontsize=14, loc='lower left')
    plt.grid(alpha=0.3)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    plt.savefig('data/data.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def load_data(n=100, centers=[[1, .3],[.3, 1]],
              std=.20, seed=123, plot=True, save=True):
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
        if el == 0:
            Y.append([1, 0])
        else:
            Y.append([0, 1])
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

    return acc, brier


# def training_set(X, Y, n=4):
#     ix_y1 = np.random.choice(np.where(Y == 1)[0], int(n / 2), replace=False)
#     ix_y0 = np.random.choice(np.where(Y == 0)[0], int(n / 2), replace=False)

#     X_data = np.concatenate([X[ix_y1], X[ix_y0]])

#     for i in range(len(X_data)):
#         X_data[i] = normalize_custom(X_data[i])

#     Y_vector = label_to_array(Y)
#     Y_data = np.concatenate([Y_vector[ix_y1], Y_vector[ix_y0]])

#     return X_data, Y_data


def training_set(X, Y, n=4, seed=123):
    np.random.seed(seed)
    ix_y1 = np.random.choice(np.where(Y == 1)[0], int(n / 2), replace=False)
    ix_y0 = np.random.choice(np.where(Y == 0)[0], int(n / 2), replace=False)

    X_data = np.concatenate([X[ix_y1], X[ix_y0]])
    X_data_new = []

    for i in range(len(X_data)):
        X_data_new.append(normalize_custom(X_data[i]))

    X_data_new = np.array(X_data_new)

    Y_vector = label_to_array(Y)
    Y_data = np.concatenate([Y_vector[ix_y1], Y_vector[ix_y0]])

    return X_data_new, Y_data



# Define the cosine classifier
def cosine_classifier(x,y):
    return 1/2 + (cosine_similarity([x], [y])**2)/2


def avg_vs_ensemble(avg, ens, ens_real=None):
    if ens_real!=None:
        plt.plot(np.arange(N_runs), p1_ens_real, marker='o', color='lightblue', label='qEnsemble (Rd)')
    N_runs = len(avg)
    plt.plot(np.arange(N_runs), ens, marker='o', color='orange', label = 'qEnsemble (QASM)')
    plt.scatter(np.arange(N_runs), avg, label='Simple AVG', color='sienna', zorder=3, linewidth=.5)
    plt.title('Quantum Ensemble vs Classical Ensemble', size=12).set_position([.5, 1.05])
    plt.xlabel('runs', size=12)
    plt.ylabel(r'$P(y^{(test)}=1$', size =12)
    plt.xticks(np.arange(0, N_runs+1, 5), size = 12)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size = 12)
    plt.ylim(0,1)
    plt.grid(alpha=.3)
    plt.legend()


def quantum_cos_random_data(x, P0, P1, err):
    fig, ax = plt.subplots(1)
    ax.plot(x, P0 , lw=2, color='blue')
    ax.fill_between(x, P0 - err, P0 + err, facecolor = 'blue', label='$y_{b} = 1$', alpha=0.5)
    ax.plot(x, P1 , lw=2, color='orange')
    ax.fill_between(x, P1 - err, P1 + err, facecolor = 'orange', label='$y_{b} = 0$' , alpha=0.5)
    #ax.plot(q, mean, lw=2, label = str(np.round(err, 1)), color=color)
    #ax.fill_between(q, mean + upper, mean - lower, facecolor=color, alpha=0.5)
    #ax.set_title(r'random walkers empirical $\mu$ and $\pm \sigma$ interval')
    #ax.set_title(r'Quantum cosine classifier', size=14).set_position([.5, 1.05])
    ax.legend(loc='center lower', prop=dict(size=12))
    ax.set_xlabel('Cosine distance', size = 14)
    ax.set_ylabel('$Pr(y^{(test)} = 1)$',size = 14)
    ax.axhline(y=.5, xmin=-1, xmax=1, color = 'gray', linestyle = '--')
    ax.set_xticklabels([0, -1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00], size=14)
    #ax.set_yticklabels([0, .2, .4, 0.2, 0.3, 0.4, 0.5], size=12)
    ax.set_yticklabels([0, 0.0, .2, .4, .6, 0.8, 1.0], size=14)
    ax.grid(alpha=.3)
    #plt.savefig('../output/cos_classifier_behaviour.png', dpi = 300, bbox_inches='tight')
    plt.show()