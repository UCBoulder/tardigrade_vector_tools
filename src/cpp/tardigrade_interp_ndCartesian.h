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

#include<array>

namespace tardigradeVectorTools{

    namespace interp{

        typedef double floatType;

        class ndCartesian{

            public:

                const unsigned int _spatial_dimension;

                const floatType * const _D;

                const unsigned int _D_size;

                const unsigned int _npts;

                const unsigned int _D_cols;

                const floatType _tolr;

                const floatType _tola;

                ndCartesian( const unsigned int spatial_dimension, const floatType * const D, const unsigned int D_size, const unsigned int npts,
                             const floatType tolr = 1e-9, const floatType tola = 1e-9 );

                const std::vector< unsigned int > * getDimensions( ){ return &_dimensions; }

                const std::vector< unsigned int > * getStrides( ){ return &_strides; }

                const std::vector< unsigned int > getCurrentBounds( const std::vector< floatType > &p, const unsigned int &insize ){ return getBoundingBoxIndices( p, insize ); }

                const std::vector< floatType > * getCurrentWeights( ){ return &_current_weights; }

                floatType eval( std::vector< floatType > &p, const unsigned int col=0 );

            protected:

                std::vector< unsigned int > getDimensions( const unsigned int span, const unsigned int index = 0 );

                void setDimensions( const std::vector< unsigned int > &dimensions ){ _dimensions = dimensions; }

                void setStrides( );

                std::vector< unsigned int > getBoundingBoxIndices( const std::vector< floatType > &p, const unsigned int insize, const unsigned int index = 0 );

                std::vector< floatType > getWeights( const std::vector< floatType > &p, const std::vector< unsigned int > &current_bounds, const unsigned int insize, const unsigned int index = 0 );

                floatType interpolateFunction( const std::vector< floatType > &p, const std::vector< unsigned int > &current_bounds, const unsigned int col = 0, const unsigned int index = 0, const unsigned int offset = 0 );

                std::vector< unsigned int > _dimensions;

                std::vector< unsigned int > _strides;

                std::vector< floatType > _current_weights;

            private:

                std::array< unsigned int, 2 > getBounds( const unsigned int index, const floatType &pd, const unsigned int dim_npts );

        };

    }

}

#include "tardigrade_interp_ndCartesian.cpp"
#endif
