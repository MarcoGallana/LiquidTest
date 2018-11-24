import h5py
import numpy  as np

class WrongFormatError(Exception):

    def __init__( self, message, expected_shape, input_shape ):

        super(WrongFormatError,self).__init__(message)

        self.expected_shape =   expected_shape
        self.input_shape    =   input_shape


class H5Reader():

    def __init__( self, data_shape = None ):
        self.data_shape = data_shape

    def read( self, filename ):

        file    =   h5py.File( filename, 'r' )
        group   =   list(file.keys())[0]

        data    =   { k: np.array(file[k]) for k in group }

        self.__check_shape(data)

        return data

    def __check_shape( self, data ):

        if self.data_shape:

            for key, value in data.items():

                expected_shape  =   self.data_shape.get(key,None)

                if expected_shape:
                    if not H5Reader.__cfr_tuple(value.shape,expected_shape):

                        message =   'Unexpected ' + key + ' shape.',

                        raise WrongFormatError(message=message,expected_shape=expected_shape,input_shape=value.shape)

    @staticmethod
    def __cfr_tuple( a, b ):

        if len(a) != len(b):
            return False
        else:

            bool    =   True

            for i,elm in enumerate(a):

                if elm is not None and b[i] is not None:
                    bool    =   elm == b[i]
                    
                    if not bool:
                        return False

            return True
