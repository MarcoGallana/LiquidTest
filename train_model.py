import reader
import os
import tools
import algo
import joblib

path_data    =   'data'
path_mdl     =   'model'

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
    model   =   algo.sklearn_nn( activation = 'relu', patience = 10, max_iter = 100 )
    model.fit( Xscale, Yscale )

    #### SAVE
    filename    =   os.path.join( path_mdl, 'neural_network.sav')
    joblib.dump( model, filename )
    filename    =   os.path.join( path_mdl, 'scalerX.sav')
    joblib.dump( scalerX, filename )
    filename    =   os.path.join( path_mdl, 'scalerY.sav')
    joblib.dump( scalerY, filename )
