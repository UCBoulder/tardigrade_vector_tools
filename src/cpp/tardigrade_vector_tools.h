/**
  *****************************************************************************
  * \file tardigrade_vector_tools.h
  *****************************************************************************
  * A collection of functions and related utilities intended to help perform
  * vector operations in cpp.
  *****************************************************************************
  */

#ifndef TARDIGRADE_VECTOR_TOOLS_H
#define TARDIGRADE_VECTOR_TOOLS_H

#include<tardigrade_error_tools.h>
#include<stdio.h>
#include<iostream>
#include<stdexcept>
#include<exception>
#include<fstream>
#include<vector>
#include<map>
#include<math.h>
#include<assert.h>
#include<string.h>
#include<numeric>
#include<algorithm>
#include<functional>

#ifdef USE_EIGEN
    #include<Eigen/Dense>
#endif

//Operator overloading
template<typename T>
std::vector<T>& operator+=(std::vector<T> &lhs, const std::vector<T> &rhs);

template<typename T>
std::vector<T>& operator+=(std::vector<T> &lhs, const T &rhs);

template<typename T>
std::vector<T> operator+(std::vector<T> lhs, const std::vector<T> &rhs);

template<typename T>
std::vector<T> operator+(std::vector<T> lhs, const T &rhs);

template<typename T>
std::vector<T> operator+(const T &lhs, std::vector<T> rhs);

template<typename T>
std::vector< std::vector< T > >& operator+=(std::vector< std::vector< T > > &lhs, const std::vector< std::vector< T > > &rhs);

template<typename T>
std::vector< std::vector< T > > operator+(std::vector< std::vector< T > > lhs, const std::vector< std::vector< T > > &rhs);

template<typename T>
std::vector<T> operator-(std::vector<T> v);

template<typename T>
std::vector< std::vector< T > > operator-(std::vector< std::vector< T > > v);

template<typename T>
std::vector<T>& operator-=(std::vector<T> &lhs, const std::vector<T> &rhs);

template<typename T>
std::vector< std::vector < T > >& operator-=(std::vector< std::vector< T > > &lhs, const std::vector< std::vector< T > > &rhs);

template<typename T>
std::vector<T>& operator-=(std::vector<T> &lhs, const T &rhs);

template<typename T>
std::vector<T> operator-(std::vector<T> lhs, const std::vector<T> &rhs);

template<typename T>
std::vector< std::vector< T > > operator-(std::vector< std::vector< T > > lhs, const std::vector< std::vector< T > > &rhs);

template<typename T>
std::vector<T> operator-(std::vector<T> lhs, const T &rhs);

template<typename T>
std::vector<T> operator-(const T &lhs, std::vector<T> rhs);

template<typename T, typename t>
std::vector<T>& operator*=(std::vector<T> &lhs, const t rhs);

template<typename T, typename t>
std::vector<T> operator*(const t lhs, std::vector<T> rhs);

template<typename T, typename t>
std::vector<T> operator*(std::vector<T> lhs, const t rhs);

template<typename T, typename t>
std::vector<T>& operator/=(std::vector<T> &lhs, const t rhs);

template<typename T, typename t>
std::vector<T> operator/(std::vector<T> lhs, const t rhs);

namespace tardigradeVectorTools{
    //Type definitions

    /** Definition of the standard size type */
    typedef unsigned int size_type;

    //Computation Utilities
    template<typename T, class M_in, class v_out>
    void computeRowMajorMean(const M_in &A_begin, const M_in &A_end, v_out v_begin, v_out v_end);

    template<class M_in, class v_out>
    void computeMean(const M_in &A_begin, const M_in &A_end, v_out v_begin, v_out v_end);

    template<typename T>
    int computeMean(const std::vector<std::vector< T > > &A, std::vector< T > &v);

    template<typename T>
    std::vector< T > computeMean(const std::vector< std::vector< T > > &A);

    template<typename T>
    int cross(const std::vector< T > &a, const std::vector< T > &b, std::vector< T > &c);

    template<typename T>
    std::vector< T > cross(const std::vector< T > &a, const std::vector< T > &b);

    template<typename T>
    int dot(const std::vector< T > &a, const std::vector< T > &b, T &c);

    template<typename T>
    T dot(const std::vector< T > &a, const std::vector< T > &b);

    template<typename T>
    std::vector< T > dot(const std::vector< std::vector< T > > &A, const std::vector< T > &b);

    template<typename T>
    std::vector< T > Tdot(const std::vector< std::vector< T > > &A, const std::vector< T > &b);

    template<typename T>
    std::vector< std::vector< T > > dot(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B);

    template<typename T>
    std::vector< std::vector< T > > dotT(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B);

    template<typename T>
    std::vector< std::vector< T > > Tdot(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B);

    template<typename T>
    std::vector< std::vector< T > > TdotT(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B);

    template<typename T>
    int inner(const std::vector< T > &A, const std::vector< T > &B, T &result);

    template<typename T>
    T inner(const std::vector< T > &A, const std::vector< T > &B);

    template<typename T>
    int inner(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B, T &result);

    template<typename T>
    T inner(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B);

    template<typename T>
    int trace(const std::vector< T > &A, T &v);

    template<typename T>
    T trace(const std::vector< T > &A);

    template<typename T>
    int trace(const std::vector< std::vector< T > > &A, T &v);

    template<typename T>
    T trace(const std::vector< std::vector< T > > &A);

    template<typename T>
    double l2norm(const std::vector< T > &v);

    template<typename T>
    double l2norm(const std::vector< std::vector< T > > &A);

    template<typename T>
    std::vector< double > unitVector(const std::vector< T > &v);

    template<typename T>
    std::vector< std::vector< T > > dyadic(const std::vector< T > &a, const std::vector< T > &b);

    template<typename T>
    int dyadic(const std::vector< T > &a, const std::vector< T > &b, std::vector< std::vector< T > > &A);

    template<typename T>
    int eye(std::vector< T > &I);

    template<typename T>
    std::vector< std::vector< T > > eye(const unsigned int dim);

    template<typename T>
    int eye(const unsigned int dim, std::vector< std::vector< T > > &I);

    template<typename T>
    T median(const std::vector< T > &x);

    template<typename T>
    std::vector< T > abs(const std::vector< T > &x);

    //Comparison Tools
    template<typename T>
    bool fuzzyEquals(const T &a, const T &b, double tolr=1e-6, double tola=1e-6);

    template<typename T>
    bool fuzzyEquals(const std::vector< T > &a, const std::vector< T > &b, double tolr=1e-6, double tola=1e-6);

    template<typename T>
    bool fuzzyEquals(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B,
        double tolr=1e-6, double tola=1e-6);

    template<typename T>
    bool equals(const T &a, const T &b);

    template<typename T>
    bool equals(const std::vector< T > &a, const std::vector< T > &b);

    template<typename T>
    bool equals(const std::vector< std::vector< T > > &a, const std::vector< std::vector< T > > &b);

    template<typename T>
    bool isParallel( const std::vector< T > &v1, const std::vector< T > &v2 );

    template<typename T>
    bool isOrthogonal( const std::vector< T > &v1, const std::vector< T > &v2 );

    template<typename T>
    void verifyOrthogonal( const std::vector< T > &v1, const std::vector< T > &v2,
                           std::string message = "Vectors are not orthogonal" );

    template<typename T>
    void verifyLength( const std::vector< T > &verifyVector, const unsigned int &expectedLength,
                       std::string message = "Vector does not have expected length" );
    template<typename T>
    void verifyLength( const std::vector< T > &verifyVectorOne,
                       const std::vector< T > &verifyVectorTwo,
                       std::string message = "Vector lengths do not match" );
    template<typename T>
    void verifyLength( const std::vector< std::vector< T > > &verifyVectorOne,
                       const std::vector< std::vector< T > > &verifyVectorTwo,
                       std::string message = "Vector lengths do not match" );

    //Access Utilities
    template <typename T>
    int getValuesByIndex(const std::vector< T > &v, const std::vector< size_type > &indices,
        std::vector< T > &subv);

    template <typename T>
    std::vector< T > getRow( const std::vector< T > &A, const unsigned int rows, const unsigned int cols, const unsigned int row );

    template <typename T>
    std::vector< T > getCol( const std::vector< T > &A, const unsigned int rows, const unsigned int cols, const unsigned int col );

    //Appending utilities
    template<typename T>
    std::vector< T > appendVectors(const std::vector< std::vector< T > > &A);

    template<typename T>
    std::vector< T > appendVectors(const std::initializer_list< std::vector< T > > &list);

    template< typename T >
    std::vector< std::vector< T > > inflate( const std::vector< T > &Avec, const unsigned int &nrows, const unsigned int &ncols );

    //Sorting utilities
    template <typename T>
    std::vector< size_type > argsort(const std::vector< T > &v);

    //Printing Utilities
    template<typename T>
    int print(const std::vector< T > &v);

    template<typename T>
    int print(const std::vector< std::vector< T > > &A);

    template<typename T>
    int rotationMatrix( const std::vector< T > &bungeEulerAngles, std::vector< T > &rotationMatrix );

    template<typename T>
    int rotationMatrix( const std::vector< T > &bungeEulerAngles, std::vector < std::vector< T > > &directionCosines );

    template<typename T>
    int rotationMatrix( const std::vector< T > &bungeEulerAngles, std::vector < std::vector< T > > &directionCosines,
                        std::vector< std::vector< T > > &dDirectionCosinesdAlpha,
                        std::vector< std::vector< T > > &dDirectionCosinesdBeta,
                        std::vector< std::vector< T > > &dDirectionCosinesdGamma );

    template<typename T>
    void computeMatrixExponential( const std::vector< T > &A, const unsigned int &dim, std::vector< T > &expA, const unsigned int nmax=40, double tola=1e-9, double tolr=1e-9);

    template<typename T>
    void computeMatrixExponential( const std::vector< T > &A, const unsigned int &dim, std::vector< T > &expA, std::vector< T > & dExpAdA, const unsigned int nmax=40, double tola=1e-9, double tolr=1e-9);

    template<typename T>
    void computeMatrixExponentialScalingAndSquaring( const std::vector< T > &A, const unsigned int &dim, std::vector< T > &expA, const unsigned int nmax=40, double tola=1e-9, double tolr=1e-9);

    template<typename T>
    void computeMatrixExponentialScalingAndSquaring( const std::vector< T > &A, const unsigned int &dim, std::vector< T > &expA, std::vector< T > & dExpAdA, const unsigned int nmax=40, double tola=1e-9, double tolr=1e-9);

    //Utilities which require Eigen
    #ifdef USE_EIGEN
        //Eigen specific type definitions
        template< typename T >
        using solverType = Eigen::ColPivHouseholderQR< Eigen::Matrix< T, -1, -1, Eigen::RowMajor > >; //!Define the matrix solver

        template<typename T>
        std::vector< double > solveLinearSystem( const std::vector< std::vector< T > > &A, const std::vector< T > &b, unsigned int &rank );

        template<typename T>
        std::vector< double > solveLinearSystem( const std::vector< std::vector< T > > &A, const std::vector< T > &b, unsigned int &rank,
                                                solverType< T > &linearSolver );

        template<typename T>
        std::vector< double > solveLinearSystem( const std::vector< T > &A, const std::vector< T > &b,
            const unsigned int nrows, const unsigned int ncols, unsigned int &rank);

        template<typename T>
        std::vector< double > solveLinearSystem( const std::vector< T > &A, const std::vector< T > &b,
            const unsigned int nrows, const unsigned int ncols, unsigned int &rank, solverType< T > &linearSolver );

        template<typename T>
        T determinant(const std::vector< T > &Avec, const unsigned int nrows, const unsigned int ncols);

        template<typename T>
        std::vector< double > inverse(const std::vector< T > &Avec, const unsigned int nrows, const unsigned int ncols);

        template<typename T>
        std::vector< std::vector< double > > inverse( const std::vector< std::vector< T > > &A );

        template<typename T>
        std::vector< double > computeFlatDInvADA( const std::vector< T > &invA, const unsigned int nrows, const unsigned int ncols );

        template<typename T>
        std::vector< std::vector< double > > computeDInvADA( const std::vector< T > &invA, const unsigned int nrows, const unsigned int ncols );

        template<typename T>
        std::vector< double > computeDDetADA(const std::vector< T > &Avec, const unsigned int nrows, const unsigned int ncols);

        template< typename T >
        std::vector< T > matrixMultiply(const std::vector< T > &A, const std::vector< T > &B,
                                             const unsigned int Arows, const unsigned int Acols,
                                             const unsigned int Brows, const unsigned int Bcols,
                                             const bool Atranspose = false, const bool Btranspose = false);

        template< typename T >
        int __matrixSqrtResidual(const std::vector< T > &A, const unsigned int Arows,
                                 const std::vector< T > &X,
                                 std::vector< double > &R, std::vector< std::vector< double > > &J);

        template< typename T >
        std::vector< double > matrixSqrt(const std::vector< T > &A, const unsigned int Arows,
                                         const double tolr = 1e-9, const double tola = 1e-9, const unsigned int maxIter = 20,
                                         const unsigned int maxLS = 5);

        template< typename T >
        std::vector< double > matrixSqrt(const std::vector< T > &A, const unsigned int Arows,
                                         std::vector< std::vector< double > > &dAdX,
                                         const double tolr = 1e-9, const double tola = 1e-9, const unsigned int maxIter = 20,
                                         const unsigned int maxLS = 5);

        template< typename T >
        void svd( const std::vector< T > &A, std::vector< std::vector< double > > &U, std::vector< double > &Sigma,
                  std::vector< std::vector< double > > &V );

        template< typename T >
        void polar_decomposition( const std::vector< T > &A, const unsigned int nrows, const unsigned int ncols,
                                  std::vector< double > &R, std::vector< double > &U, const bool left = false );

    #endif

}

#include "tardigrade_vector_tools.cpp"
#endif
