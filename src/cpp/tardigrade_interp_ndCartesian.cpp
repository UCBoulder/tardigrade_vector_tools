
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

#include<tardigrade_error_tools.h>

namespace tardigradeVectorTools{

    namespace interp{

        ndCartesian::ndCartesian( const unsigned int spatial_dimension, const floatType * const D, const unsigned int D_size, const unsigned int npts,
                                  const floatType tolr, const floatType tola ) : _spatial_dimension( spatial_dimension ), _D( D ), _D_size( D_size ), _npts( npts ), _D_cols( _D_size / _npts ), _tolr( tolr ), _tola( tola ){
            /*!
             * The constructor function for ndCartesian
             */

            TARDIGRADE_ERROR_TOOLS_CHECK( _D_cols * _npts == _D_size, "The size of D ( " + std::to_string( _D_size ) + " ) is not a scalar multiple of the number of points ( " + std::to_string( _npts ) + " )" );

            TARDIGRADE_ERROR_TOOLS_CHECK( _D_cols >= _spatial_dimension, "The spatial dimension ( " + std::to_string( _spatial_dimension ) + " ) is larger than the number of columns in D ( " + std::to_string( _D_cols ) );

            // Set the dimensions of the cartesian grid
            setDimensions( getDimensions( _npts ) );

            // Set the strides for the dimensions
            setStrides( );

        }

        std::vector< unsigned int > ndCartesian::getDimensions( const unsigned int span, const unsigned int index ){
            /*!
             * Get the number of points in each of the spatial dimensions
             * 
             * \param span: The number of points to search in D
             * \param index: The index of the spatial dimension to search
             */

            const floatType tol = _tolr * std::fabs( *( _D + index ) ) + _tola;

            unsigned int n_remainder = 0;

            for ( unsigned int n = 0; n < span; n++ ){

                if ( std::fabs( *( _D + _D_cols * n + index ) - *( _D + index ) ) > tol ){

                    break;

                }

                n_remainder++;

            }

            std::vector< unsigned int > nd = { span / n_remainder };

            if ( ( index + 1 ) < _spatial_dimension ){

                std::vector< unsigned int > nd_sub = getDimensions( n_remainder, index + 1 );

                nd.insert( nd.end( ), nd_sub.begin( ), nd_sub.end( ) );

            }

            return nd;

        }

        void ndCartesian::setStrides( ){
            /*!
             * Set the strides for each of the cartesian dimensions
             */

            _strides = std::vector< unsigned int >( _dimensions.size( ), 0 );

            unsigned int ndim = _dimensions.size( );

            for ( unsigned int d = 0; d < ndim; d++ ){

                _strides[ d ] = 1;

                for ( unsigned int i = d + 1; i < ndim; i++ ){

                    _strides[ d ] *= _dimensions[ i ];

                }

            }

        }

    }

}
