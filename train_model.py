import reader
import os
import tools
import liquid_algos as algo
import joblib

path_data    =   'data'
path_mdl     =   'model'

if __name__ == '__main__':

    ### read data
    filenameX    =   os.path.join( path_data, 'X.h5' )
    filenameY    =   os.path.join( path_data, 'Y.h5' )

    r   =   reader.H5Reader( data_shape = {'Y':(None,5)} )

    dataX    =   r.read( filenameX )
    dataY    =   r.read( filenameY )

    X   =   dataX['X']
    Y   =   dataY['Y']

    # train NN
    scalerX, scalerY  = algo.fit_scaler( X ), algo.fit_scaler( Y )
    Xscale, Yscale    = scalerX.transform(X), scalerY.transform(Y)
    model   =   algo.sklearn_nn( hidden_layer_sizes = (256,128), activation = 'relu', max_iter = 100 )
    model.fit( Xscale, Yscale )

    #### save NN
    filename    =   os.path.join( path_mdl, 'neural_network.sav')
    joblib.dump( model, filename )
    filename    =   os.path.join( path_mdl, 'scalerX.sav')
    joblib.dump( scalerX, filename )
    filename    =   os.path.join( path_mdl, 'scalerY.sav')
    joblib.dump( scalerY, filename )

    ### train LinearRegression
    model   =   algo.linreg()
    model.fit( X, Y )
    # save LinearRegression
    filename    =   os.path.join( path_mdl, 'linreg.sav')
    joblib.dump( model, filename )
