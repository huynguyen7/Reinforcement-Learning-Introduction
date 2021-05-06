import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def visualize_figure_5_1(V_usable_ace_10k, V_no_usable_ace_10k, V_usable_ace_500k, V_no_usable_ace_500k, show=False, save=False):
    matplotlib.rc('font', size=6)

    ax = plt.subplot(2,2,1)
    plt.title('10k EPS WITH USABLE ACE')
    plt.xticks(ticks=np.arange(0,10,1), labels=np.arange(1,10+1,1))
    plt.yticks(ticks=np.arange(0,10,1), labels=np.arange(12,21+1,1))
    plt.imshow(V_usable_ace_10k, cmap='viridis', interpolation='nearest')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    plt.subplot(2,2,2)
    plt.title('500k EPS WITH USABLE ACE')
    plt.xticks(ticks=np.arange(0,10,1), labels=np.arange(1,10+1,1))
    plt.yticks(ticks=np.arange(0,10,1), labels=np.arange(12,21+1,1))
    plt.imshow(V_usable_ace_500k, cmap='viridis', interpolation='nearest')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    plt.subplot(2,2,3, sharex=ax, sharey=ax)
    plt.title('10k EPS WITHOUT USABLE ACE')
    plt.xticks(ticks=np.arange(0,10,1), labels=np.arange(1,10+1,1))
    plt.yticks(ticks=np.arange(0,10,1), labels=np.arange(12,21+1,1))
    plt.imshow(V_no_usable_ace_10k, cmap='viridis', interpolation='nearest')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.xlabel("DEALER'S UPCARD")
    plt.ylabel("PLAYER'S SUM")

    plt.subplot(2,2,4, sharex=ax, sharey=ax)
    plt.title('500k EPS WITHOUT USABLE ACE')
    plt.xticks(ticks=np.arange(0,10,1), labels=np.arange(1,10+1,1))
    plt.yticks(ticks=np.arange(0,10,1), labels=np.arange(12,21+1,1))
    heatmap = plt.imshow(V_no_usable_ace_500k, cmap='viridis', interpolation='nearest')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.colorbar(heatmap)

    if show:
        plt.show()
    if save:
        plt.savefig('./5_1.png')
        plt.close()


def visualize_figure_5_2(usable_ace_optimal_V, usable_ace_optimal_pi, no_usable_ace_optimal_V, no_usable_ace_optimal_pi, show=False, save=False):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rc('font', size=6)

    plt.title('500k EPISODES')
    ax = plt.subplot(2,2,1)
    plt.title('USABLE ACE OPTIMAL PI')
    plt.xticks(ticks=np.arange(0,10,1), labels=np.arange(1,10+1,1))
    plt.yticks(ticks=np.arange(0,10,1), labels=np.arange(12,21+1,1))
    heatmap = plt.imshow(usable_ace_optimal_pi, cmap='viridis', interpolation='nearest')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.colorbar(heatmap)

    plt.subplot(2,2,2, sharex=ax, sharey=ax)
    plt.title('USABLE ACE OPTIMAL V')
    plt.xticks(ticks=np.arange(0,10,1), labels=np.arange(1,10+1,1))
    plt.yticks(ticks=np.arange(0,10,1), labels=np.arange(12,21+1,1))
    heatmap = plt.imshow(usable_ace_optimal_V, cmap='viridis', interpolation='nearest')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.colorbar(heatmap)

    plt.subplot(2,2,3, sharex=ax, sharey=ax)
    plt.title('NO USABLE ACE OPTIMAL PI')
    plt.xticks(ticks=np.arange(0,10,1), labels=np.arange(1,10+1,1))
    plt.yticks(ticks=np.arange(0,10,1), labels=np.arange(12,21+1,1))
    heatmap = plt.imshow(no_usable_ace_optimal_pi, cmap='viridis', interpolation='nearest')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.colorbar(heatmap)
    plt.xlabel("DEALER'S UPCARD")
    plt.ylabel("PLAYER'S SUM")

    plt.subplot(2,2,4, sharex=ax, sharey=ax)
    plt.title('NO USABLE ACE OPTIMAL V')
    plt.xticks(ticks=np.arange(0,10,1), labels=np.arange(1,10+1,1))
    plt.yticks(ticks=np.arange(0,10,1), labels=np.arange(12,21+1,1))
    heatmap = plt.imshow(no_usable_ace_optimal_V, cmap='viridis', interpolation='nearest')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.colorbar(heatmap)

    if show:
        plt.show()
    if save:
        plt.savefig('./5_2.png')
        plt.close()

