from    sklearn.linear_model    import  LinearRegression
from    sklearn.neural_network  import  MLPRegressor
from    sklearn.preprocessing   import  MinMaxScaler


def linreg( intercept = False ):

    model   =   LinearRegression(   fit_intercept   =   intercept )

    return model

def sklearn_nn( hidden_layer_sizes  = (256,), activation = 'relu', validation_fraction = .25, max_iter = 200 ):

    model   =   MLPRegressor(   hidden_layer_sizes  =   hidden_layer_sizes,
                                solver              =   'adam',
                                activation          =   activation,
                                learning_rate       =   'adaptive',
                                early_stopping      =   True,
                                validation_fraction =   validation_fraction,
                                max_iter            =   max_iter,
                                verbose             =   True,
                                random_state        =   0 )

    return model

def fit_scaler( X ):
    scaler  =   MinMaxScaler()
    return  scaler.fit(X)
