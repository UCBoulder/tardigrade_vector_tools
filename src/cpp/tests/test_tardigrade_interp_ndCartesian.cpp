/**
  * \file test_tardigrade_interp_ndCartesian.cpp
  *
  * Tests for tardigrade_interp_ndCartesian
  */

#include<vector>
#include<iostream>
#include<fstream>
#include<math.h>
#define USE_EIGEN
#include<tardigrade_interp_ndCartesian.h>

#define BOOST_TEST_MODULE test_tardigrade_interpNdCartesian
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#define DEFAULT_TEST_TOLERANCE 1e-6
#define CHECK_PER_ELEMENT boost::test_tools::per_element( )

typedef double floatType;
typedef std::vector< floatType > vectorType;
typedef std::vector< vectorType > matrixType;

void print( vectorType a ){
    /*!
     * Print the vector to the terminal
     */

    for ( unsigned int i=0; i<a.size( ); i++ ){
        std::cout << a[ i ] << " ";
    }
    std::cout << "\n";
}

void print( matrixType A ){
    /*!
     * Print the matrix to the terminal
     */

    for ( unsigned int i=0; i<A.size( ); i++ ){
        print( A[ i ] );
    }
}

BOOST_AUTO_TEST_CASE( test_getDimensions, * boost::unit_test::tolerance( DEFAULT_TEST_TOLERANCE ) ){
    /*!
     * Test the function that extracts the size of each of the interpolation dimensions
     */

    BOOST_TEST( true );

}
