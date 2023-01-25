from gametheory.regretmatching import RegretMatchingStrategies, RegretMatchingActions
import gametheory.game as game 
from utils.blitmanager import BlitManager

import matplotlib.pyplot as plt
import numpy as np


def plot(gp):
    # set up plots
    plotfreq = 1
    axes = {}
    lns = {}
    y = {}
    fig, (ax_freq1, axes['ce'], axes['cce']) = plt.subplots(3, 1)

    # set up game
    p, rext, rint, s = next(gp)

    # set up CCE/CE/freq
    y['cce'] = np.ones(plotfreq) * -1
    y['ce'] = np.ones(plotfreq) * -1
    y['cce'][0] = rext
    y['ce'][0] = rint
    for j in range(p.size):
        y[f'f{j}1'] = np.ones(plotfreq) * -1
        y[f'f{j}1'][0] = p[j]
        axes[f'f{j}1'] = ax_freq1
    for k in y.keys():
        lns[k] = axes[k].plot(np.arange(plotfreq)+1, y[k], '-', alpha=0.7)[0]
        axes[k].axhline(alpha=0)
    ax_freq1.set_ylim(0,1)

    # plot first points
    bm = BlitManager(fig.canvas, lns.values())
    plt.show(block=False)
    plt.pause(0.1)

    i = 1
    redrawtime = plotfreq
    while True:
        # get data
        p, rext, rint, s = next(gp)

        # make data and axes longer if necessary
        if i % redrawtime == 0:
            plotfreq *= 2
            redrawtime = i + plotfreq
            xs = np.arange(redrawtime)
            for k in y.keys():
                lns[k].set_xdata(xs)
                y[k] = np.hstack((y[k], np.zeros(plotfreq)/0))
                axes[k].set_xlim(right=xs[-1])
                if k[0] != 'f':
                    mx = max(y[k][i//5:i].max(), 0)
                    mn = min(y[k][i//5:i].min(), 0)
                    mx += 0.1*(mx-mn)
                    mn -= 0.1*(mx-mn)
                    axes[k].set_ylim(mn, mx)
                    axes[k].lines.pop(-1)
                    axes[k].axhline(color='black', alpha=0.4)
            plt.draw()


        # update data
        y['cce'][i] = rext
        y['ce'][i] = rint
        for j in range(p.size):
            y[f'f{j}1'][i] = s[j]
        
        # update graphs
        fpd = 100
        if  i % redrawtime == 0 or (plotfreq > fpd and (i - redrawtime) % (plotfreq // fpd) == 0) or (plotfreq > 10000 and (i - redrawtime) % 100 == 0):
            for k in y.keys():
                lns[k].set_ydata(y[k])
            bm.update()
        i += 1


def main():
    g = game.shapley(RegretMatchingStrategies)
    plot(g.selfplay())

if __name__ == "__main__":
    main()