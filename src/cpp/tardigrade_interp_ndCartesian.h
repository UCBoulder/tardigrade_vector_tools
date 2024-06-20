/**
  *****************************************************************************
  * \file tardigrade_interp_NdCartesian.h
  *****************************************************************************
  * A class which allows for interpolation of Nd functions which have values
  * defined in a cartesian grid as (p1_x, p1_y, ... p1_n, v1_1, v1_2, ...
  * p1_x, p1_y, ... p2_n, v2_1, v2_2, ...)
  *****************************************************************************
  */

#ifndef TARDIGRADE_INTERP_NDCARTESIAN_H
#define TARDIGRADE_INTERP_NDCARTESIAN_H

#include<tardigrade_error_tools.h>

namespace tardigradeVectorTools{

    namespace interp{

        typedef double floatType;

        class ndCartesian{

            public:

                const unsigned int _spatial_dimension;

                const floatType * const _D;

                const unsigned int _npts;

                const floatType _tolr;

                const floatType _tola;

                ndCartesian( const unsigned int spatial_dimension, const floatType * const D, const unsigned int npts,
                             const floatType tolr = 1e-9, const floatType tola = 1e-9 );

        };

    }

}

#include "tardigrade_interp_ndCartesian.cpp"
#endif
