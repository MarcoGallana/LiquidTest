import keras

from    keras.models            import  Sequential
from    keras.layers            import  Dense
from    keras.layers            import  Dropout
from    sklearn.ensemble        import  RandomForestRegressor
from    sklearn.linear_model    import  BayesianRidge
from    sklearn.linear_model    import  Lasso
from    sklearn.neural_network  import  MLPRegressor
from    sklearn.preprocessing   import  MinMaxScaler

def nn( input_shape, output_shape ):

    input   =   Dense(  units       =   input_shape,
                        input_shape =   (input_shape,),
                        activation  =   'relu'
                        )
    hidden_1  =   Dense(    units      =   256,
                            activation =   'sigmoid',
                        )
    hidden_2  =   Dense(    units      =   256,
                            activation =   'sigmoid',
                        )
    output  =   Dense(  units      =   output_shape,
                        activation =   'linear',
                        )

    model   =   Sequential()
    model.add( input )
    model.add( Dropout(.5,seed=0) )
    model.add( hidden_1 )
    model.add( Dropout(.5,seed=0) )
    model.add( output )

    model.compile(
            loss	   =	'mean_squared_error',
            metrics	   =	['mae'],
            optimizer  =	'adam',
        )

    return model

def linear_nn( input_shape, output_shape ):


    output  =   Dense(  input_shape =   (input_shape,),
                        units       =   output_shape,
                        activation  =   'linear',
                        )

    model   =   Sequential()
    model.add( output )

    model.compile(
            loss	   =	'mean_squared_error',
            metrics	   =	['mae'],
            optimizer  =	'adam',
        )

    return model

def random_forest( n_estimators = 100 ):

    model   =   RandomForestRegressor(  n_estimators    =   n_estimators,
                                        random_state    =   0,
                                        warm_start      =   True,
                                        verbose         =   1 )

    return model

def bayesian_ridge( n_iter  = 300, intercept = False ):

    model   =   BayesianRidge(  n_iter          =   n_iter ,
                                fit_intercept   =   intercept,
                                verbose         =   1 )

    return model

def lasso( alpha  = 0.001, intercept = False ):

    model   =   Lasso(  alpha           =   alpha ,
                        fit_intercept   =   intercept,
                        precompute      =   True,
                        random_state    =   0 )

    return model

def sklearn_nn( hidden_layer_sizes  = (256,), activation = 'relu', validation_fraction = .25, patience = 20, max_iter = 200 ):

    model   =   MLPRegressor(   hidden_layer_sizes  =   hidden_layer_sizes,
                                solver              =   'adam',
                                activation          =   activation,
                                learning_rate       =   'adaptive',
                                early_stopping      =   True,
                                validation_fraction =   validation_fraction,
                                n_iter_no_change    =   patience,
                                max_iter            =   max_iter,
                                verbose             =   True,
                                random_state        =   0 )

    return model

def fit_scaler( X ):
    scaler  =   MinMaxScaler()
    return  scaler.fit(X)
