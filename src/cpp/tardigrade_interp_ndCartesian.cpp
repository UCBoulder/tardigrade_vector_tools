
/**
  *****************************************************************************
  * \file tardigrade_interp_ndCartesian.cpp
  *****************************************************************************
  * A class which allows for interpolation of Nd functions which have values
  * defined in a cartesian grid as (p1_x, p1_y, ... p1_n, v1_1, v1_2, ...
  * p1_x, p1_y, ... p2_n, v2_1, v2_2, ...)
  *****************************************************************************
  */

#include "tardigrade_interp_ndCartesian.h"

namespace tardigradeVectorTools{

    namespace interp{

        ndCartesian::ndCartesian( const unsigned int spatial_dimension, const floatType * const D, const unsigned int npts,
                                  const floatType tolr, const floatType tola ) : _spatial_dimension( spatial_dimension ), _D( D ), _npts( npts ), _tolr( tolr ), _tola( tola ){
            /*!
             * The constructor function for ndCartesian
             */

        }

    }

}
