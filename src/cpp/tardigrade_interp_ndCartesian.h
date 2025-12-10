/**
  *****************************************************************************
  * \file tardigrade_interp_ndCartesian.h
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

        typedef double floatType; //!< The floating point type used in the interpolation

        //! A class which performs n-dimensional interpolation on row-major ordered data
        class ndCartesian{

            public:

                const unsigned int _spatial_dimension; //!< The spatial dimension of the data

                const floatType * const _D; //!< A pointer to the start of the data array

                const unsigned int _D_size; //!< The size of the data array

                const unsigned int _npts; //!< The number of points in the data array

                const unsigned int _D_cols; //!< The number of columns in the data array

                const floatType _tolr; //!< The relative tolerance

                const floatType _tola; //!< The absolute tolerance

                ndCartesian( const unsigned int spatial_dimension, const floatType * const D, const unsigned int D_size, const unsigned int npts,
                             const floatType tolr = 1e-9, const floatType tola = 1e-9 );

                //! Get the dimensions of the data-set
                const std::vector< unsigned int > * getDimensions( ){ return &_dimensions; }

                //! Get the strides required to process the data set
                const std::vector< unsigned int > * getStrides( ){ return &_strides; }

                //! Get the indices of the current bounds
                const std::vector< unsigned int > getCurrentBounds( const std::vector< floatType > &p, const unsigned int &insize ){ return getBoundingBoxIndices( p, insize ); }

                //! Get the weights of the bounds
                const std::vector< floatType > getCurrentWeights( const std::vector< floatType > &p, const unsigned int &insize ){ std::vector< unsigned int > current_bounds = getCurrentBounds( p, insize ); return getWeights( p, current_bounds, insize ); }

                floatType eval( std::vector< floatType > &p, const unsigned int col=0 );

            protected:

                std::vector< unsigned int > getDimensions( const unsigned int span, const unsigned int index = 0 );

                void setDimensions( const std::vector< unsigned int > &dimensions ){
                    /*!
                     * Set the dimensionality of the incoming data
                     *
                     * \param &dimensions: The dimensions of the data
                     */
                    _dimensions = dimensions;
                }

                void setStrides( );

                std::vector< unsigned int > getBoundingBoxIndices( const std::vector< floatType > &p, const unsigned int insize, const unsigned int index = 0 );

                std::vector< floatType > getWeights( const std::vector< floatType > &p, const std::vector< unsigned int > &current_bounds, const unsigned int insize, const unsigned int index = 0 );

                floatType interpolateFunction( const std::vector< floatType > &p, const std::vector< unsigned int > &current_bounds, const std::vector< floatType > &current_weights,
                                               const unsigned int col = 0, const unsigned int index = 0, const unsigned int offset = 0 );

                std::vector< unsigned int > _dimensions; //!< The dimensions of the n-dimensional data

                std::vector< unsigned int > _strides; //!< The strides required to traverse the data

                std::vector< floatType > _current_weights; //!< The current weights of the surrounding points

            private:

                std::array< unsigned int, 2 > getBounds( const unsigned int index, const floatType &pd, const unsigned int dim_npts );

        };

    }

}

#include "tardigrade_interp_ndCartesian.cpp"
#endif
