import reader
import os
import tools
import algo

path    =   'data'

if __name__ == '__main__':

    filenameX    =   os.path.join( path, 'X.h5' )
    filenameY    =   os.path.join( path, 'Y.h5' )

    r   =   reader.H5Reader( data_shape = {'Y':(None,5)} )

    dataX    =   r.read( filenameX )
    dataY    =   r.read( filenameY )

    X   =   dataX['X']
    Y   =   dataY['Y']

    # keras model
    model   =   algo.nn( X.shape[1], Y.shape[1] )
    model.fit( X, Y, batch_size=100, epochs=1000 )

    Yr  =   model.predict( X )
    
    print( 'MAE in-sample: ',  tools.mae( Y, Yr ))
    print( 'MSE in-sample: ',  tools.mse( Y, Yr ))
    print( 'MAPE in-sample: ',  tools.mape( Y, Yr ))

    tools.plot( Yr )
    tools.qqplot( Yr )
    tools.boxplot( Yr )
