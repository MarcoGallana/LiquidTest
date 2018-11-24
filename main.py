import reader
import os

path    =   'data'

if __name__ == '__main__':

    filenameX    =   os.path.join( path, 'X.h5' )
    filenameY    =   os.path.join( path, 'Y.h5' )

    r   =   reader.H5Reader( data_shape = {'Y':(None,5)} )

    dataX    =   r.read( filenameX )
    dataY    =   r.read( filenameY )
