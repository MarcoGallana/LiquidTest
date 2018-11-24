import pylab
import matplotlib.pyplot    as plt
import scipy.stats          as stats

def plot( data ):

    nplot = data.shape[1]

    for i in range(0,nplot):
        plt.subplot(nplot,1,i+1)
        plt.plot( data[:,i] )

    plt.show(block=True)

def qqplot( data ):

    nplot = data.shape[1]

    for i in range(0,nplot):
        plt.subplot(nplot,1,i+1)
        stats.probplot(data[:,i], dist="norm", plot=pylab)

    plt.show(block=True)

def boxplot( data ):

    plt.boxplot( data )
    plt.show(block=True)
