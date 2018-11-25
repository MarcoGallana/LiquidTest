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

    #
    scalerX, scalerY  = algo.fit_scaler( X ), algo.fit_scaler( Y )
    Xscale, Yscale    = scalerX.transform(X), scalerY.transform(Y)
    model   =   algo.sklearn_nn( activation = 'relu' )
    model.fit( Xscale, Yscale )

    # bayesian regression
    # model   =   algo.create_model_linear( X.shape[1], Y.shape[1] )
    # model.fit( X, Y, batch_size=100, epochs=1000 )

    Yr  =   model.predict( Xscale )
    Yr  =   scalerY.inverse_transform( Yr )

    print( 'MAE in-sample: ',  tools.mae( Y, Yr ))
    print( 'MSE in-sample: ',  tools.mse( Y, Yr ))
    print( 'MAPE in-sample: ',  tools.mape( Y, Yr ))

    tools.plot( Yr )
    tools.QQplot( Yr, Y )
    tools.boxplot( Yr )
