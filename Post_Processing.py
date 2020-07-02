from Utils import *
from modeling import *

data = pd.read_csv('output/results_ensemble.csv',
                   names = ['N', 'n_train', 'n_swap', 'd', 'balanced',
                            'test_size', 'dev', 'accuracy', 'brier', 'seed'])
data['B']=2**data.d
ctrl_size = data.d.unique()
std_all = data.dev.unique()
data = data[data.dev!=.2]

data = data.drop_duplicates()


# Plot distribution of the accuracy

ax = sns.boxplot(x="dev", y="accuracy", hue="d", data=data)
plt.xlabel('Standard deviation', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(alpha=.3)
plt.gca().get_legend().remove()
plt.savefig('output/results_accuracy.png', bbox_inches='tight')
plt.show()


ax = sns.boxplot(x="dev", y="brier", hue="B", data=data)
plt.grid(alpha=.3)
plt.xlabel('Standard deviation', fontsize=15)
plt.ylabel('Brier Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks([.10, .15, .20, .25, .30],fontsize=15)
plt.gca().get_legend().remove()
plt.savefig('output/results_brier.png', bbox_inches='tight')
plt.show()


# ax = sns.boxplot(x="dev", y="brier", hue="d", data=data)
fig = plt.figure(figsize=(6, 1))
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='center', ncol=4, title='Ensemble size')
plt.savefig('output/legend_performance.png', dpi=150, bbox_inches='tight')
plt.show()


## Quantum Ensemble as Simple Averaging - Predictions (Section 5.2)


data_sim = pd.read_csv('notebooks/output/sim_results.csv')
out_sim = data_sim.values.tolist()

def plot_cls(predictions,
             #labels=['$f_1$', '$f_2$', '$f_3$', '$f_4$', 'AVG', '$Ensemble$'],
             title='Test point classification',
             file='ens_vs_single.png'):
    fontsize=28
    N = len(predictions)
    fig, ax = plt.subplots(figsize=(12,3))
    plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars
    prob_0 = [p[0] for p in predictions]
    prob_1 = [p[1] for p in predictions]
    # label = [l['label'] for l in dictionary]
    pl1 = ax.bar(ind, prob_0, width, bottom=0, color='orange')
    pl2 = ax.bar(ind + width, prob_1, width, bottom=0, color='blue')
    ax.set_title(title, size=fontsize)#, y=0, pad=-65)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels([r'$\hat{f}_1$', r'$\hat{f}_2$', r'$\hat{f}_3$', r'$\hat{f}_4$', 'AVG', 'qEnsemble'], size=fontsize)
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],size=fontsize)
    # ax.legend((pl1[0], pl2[0]), (r'$P(\tilde{y}=0)$', r'$P(\tilde{y}=1)$'), prop=dict(size=20))
    ax.autoscale_view()
    plt.ylim(0, 1)
    #plt.xlabel('Classifier', fontsize=18)
    #plt.xlabel(r'$P(\tilde{y})')
    plt.grid(alpha=.2)
    ax.tick_params(pad=10)
    if file is not None:
        plt.savefig('output/' + file + '.png', dpi=200,  bbox_inches='tight')
    plt.show()

plot_cls(out_sim, title= 'QASM simulator', file='SIM_ens_vs_single')



data_rl = pd.read_csv('notebooks/output/rl_results.csv')
out_real = data_rl.values.tolist()


def plot_cls(predictions,
             #labels=['$f_1$', '$f_2$', '$f_3$', '$f_4$', 'AVG', '$Ensemble$'],
             title='Test point classification',
             file='ens_vs_single.png'):
    fontsize=28
    N = len(predictions)
    fig, ax = plt.subplots(figsize=(12,3))
    plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars
    prob_0 = [p[0] for p in predictions]
    prob_1 = [p[1] for p in predictions]
    # label = [l['label'] for l in dictionary]
    pl1 = ax.bar(ind, prob_0, width, bottom=0, color='orange')
    pl2 = ax.bar(ind + width, prob_1, width, bottom=0, color='blue')
    ax.set_title(title, size=fontsize)#, y=0, pad=-65)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels([r'$\hat{f}_1$', r'$\hat{f}_2$', r'$\hat{f}_3$', r'$\hat{f}_4$', 'AVG', 'qEnsemble'], size=fontsize)
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],size=fontsize)
    ax.legend((pl1[0], pl2[0]), (r'$P(y^{(test)}=0)$', r'$P(y^{(test)}=1)$'),
              prop=dict(size=fontsize), bbox_to_anchor = (.80, -0.3), ncol=2)
    ax.autoscale_view()
    plt.ylim(0, 1)
    #plt.xlabel('Classifier', fontsize=fontsize)
    #plt.xlabel(r'$P(\tilde{y})')
    plt.grid(alpha=.2)
    ax.tick_params(pad=10)
    if file is not None:
        plt.savefig('output/' + file + '.png', dpi=200,  bbox_inches='tight')
    plt.show()

plot_cls(out_real, title='Real device', file='RL_ens_vs_single')

