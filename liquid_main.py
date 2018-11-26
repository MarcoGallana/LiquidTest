import  reader
import  os
import  liquid_algos
import  argparse
import  joblib
import  h5py
import  tools

PATH_MDL    =   'model'
X_DIM       =   56
Y_DIM       =   5

if __name__ == '__main__':

    parser  =   argparse.ArgumentParser()

    parser.add_argument('-X', help = 'X.h5 file (with path)')
    parser.add_argument('--Y', help = '(optional) Y.h5 file (with path)')
    parser.add_argument('--file_out', help = '(optional) .h5 file output (with path)')

    args = parser.parse_args()

    # reading X data
    filenameX  =   args.X

    r   =   reader.H5Reader({'X':(None,X_DIM), 'Y':(None,Y_DIM)})

    dataX   =   r.read(filenameX)
    X       =   dataX['X']

    # load models
    filename    =   os.path.join( PATH_MDL, 'neural_network.sav')
    model       =   joblib.load( filename )
    filename    =   os.path.join( PATH_MDL, 'scalerX.sav')
    scalerX     =   joblib.load( filename )
    filename    =   os.path.join( PATH_MDL, 'scalerY.sav')
    scalerY     =   joblib.load( filename )
    filename    =   os.path.join( PATH_MDL, 'linreg.sav')
    benchmark   =   joblib.load( filename )

    # neural_network predict
    X_scale     =   scalerX.transform( X )
    Y_model     =   scalerY.inverse_transform( model.predict(X_scale) )

    # benchmark predict
    Y_benchmark =   benchmark.predict( X )

    if args.file_out:

        h = h5py.File(args.file_out, 'w')
        h.create_dataset('Y_model',data=Y_model)
        h.create_dataset('Y_benchmark',data=Y_benchmark)

    if args.Y:

        filenameY   =   args.Y
        dataY   =   r.read(filenameY)
        Y       =   dataY['Y']

        error_model     =   Y-Y_model
        error_benchmark =   Y-Y_benchmark

        mae_model   =   tools.mae( Y, Y_model )
        mse_model   =   tools.mse( Y, Y_model )

        mae_benchmark   =   tools.mae( Y, Y_benchmark )
        mse_benchmark   =   tools.mse( Y, Y_benchmark )

        print('MAE model: ', mae_model)
        print('MSE model: ', mse_model)
        print('MAE benchmark: ', mse_benchmark)
        print('MSE benchmark: ', mae_benchmark)

        tools.QQplot( Y_model, Y )
        tools.QQplot( Y_benchmark, Y )
