import matplotlib.pyplot    as plt
import scipy.stats          as stats
import numpy                as np

from matplotlib             import  cm
from mpl_toolkits.mplot3d   import  Axes3D
from sklearn.metrics        import  mean_squared_error, mean_absolute_error

def plot( data, shape=None ):

    shape   =   shape if shape else (data.shape[1],1)

    rows, cols    =   shape

    for i in range(0,data.shape[1]):
        plt.subplot(rows,cols,i+1)
        plt.plot( data[:,i] )

    plt.show(block=True)

def nomrQQplot( data, shape=None ):

    shape   =   shape if shape else (data.shape[1],1)

    rows, cols    =   shape

    for i in range(0,data.shape[1]):
        plt.subplot(rows,cols,i+1)
        stats.probplot(data[:,i], dist="norm", plot=plt)

    plt.show(block=True)

def QQplot( data_1, data_2, shape=None ):

    shape   =   shape if shape else (data_1.shape[1],1)

    rows, cols    =   shape

    for i in range(0,data_1.shape[1]):
        plt.subplot(rows,cols,i+1)
        d_1, d_2    =   np.sort(data_1[:,i],kind='mergesort'), np.sort(data_2[:,i],kind='mergesort')
        plt.plot( d_1, 'o' )
        plt.plot( d_2, '-r' )
        plt.legend(['data_1','data_2'])

    plt.show(block=True)

def boxplot( data ):

    plt.boxplot( data )
    plt.show(block=True)


def corrsurf( X, Y ):

    _, ny    =   Y.shape
    _, nx    =   X.shape

    z   =   np.zeros((ny,nx))

    for i in range(0,ny):
        for j in range(0,nx):
            xx  =   X[:,j].reshape(1,-1)
            yy  =   Y[:,i].reshape(1,-1)
            c   =   np.corrcoef( xx, yy )
            z[i][j]  =   np.mean(np.diag(c[::-1]))

    x, y = np.meshgrid( np.arange(0,nx), np.arange(0,ny) )

    fig     =   plt.figure()
    ax      =   fig.gca(projection='3d')
    surf    =   ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show(block=True)

def mae( Y, Yf ):
    return mean_absolute_error( Y, Yf, multioutput = 'raw_values' )

def mse( Y, Yf ):
    return mean_squared_error( Y, Yf, multioutput = 'raw_values' )

def mape( Y, Yf ):
    return mean_absolute_error( Y, Yf, multioutput = 'raw_values' )/np.mean(Y, axis=0)
