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

    template<typename T, class M_in, class v_out>
    void computeMean(const M_in &A_begin, const M_in &A_end, v_out v_begin, v_out v_end);

    template<typename T>
    int computeMean(const std::vector<std::vector< T > > &A, std::vector< T > &v);

    template<typename T>
    std::vector< T > computeMean(const std::vector< std::vector< T > > &A);

    template<class v_in, class v_out>
    int cross(const v_in &a_begin, const v_in &a_end, const v_in &b_begin, const v_in &b_end, v_out c_begin, v_out c_end);

    template<typename T>
    int cross(const std::vector< T > &a, const std::vector< T > &b, std::vector< T > &c);

    template<typename T>
    std::vector< T > cross(const std::vector< T > &a, const std::vector< T > &b);

    template<typename T>
    int dot(const std::vector< T > &a, const std::vector< T > &b, T &c);

    template<typename T>
    T dot(const std::vector< T > &a, const std::vector< T > &b);

    template<typename T, class M_in, class v_in, class v_out>
    void rowMajorDot(const M_in &A_begin, const M_in &A_end, const v_in &b_begin, const v_in &b_end, v_out c_begin, v_out c_end);

    template<typename T, class M_in, class v_in, class v_out>
    void dot(const M_in &A_begin, const M_in &A_end, const v_in &b_begin, const v_in &b_end, v_out c_begin, v_out c_end);

    template<typename T>
    std::vector< T > dot(const std::vector< std::vector< T > > &A, const std::vector< T > &b);

    template<typename T, class M_in, class v_in, class v_out>
    void Tdot( const M_in &A_begin, const M_in &A_end, const v_in &b_begin, const v_in &b_end, v_out c_begin, v_out c_end );

    template<typename T, class M_in, class v_in, class v_out>
    void rowMajorTdot( const M_in &A_begin, const M_in &A_end, const v_in &b_begin, const v_in &b_end, v_out c_begin, v_out c_end );

    template<typename T>
    std::vector< T > Tdot(const std::vector< std::vector< T > > &A, const std::vector< T > &b);

    template<typename T, class M_in, class M_out>
    void dot( const M_in &A_begin, const M_in &A_end, const M_in &B_begin, const M_in &B_end, M_out C_begin, M_out C_end );

    template<typename T, class M_in, class M_out>
    void rowMajorDot( const M_in &A_begin, const M_in &A_end, const M_in &B_begin, const M_in &B_end, const size_type rows, const size_type cols, M_out C_begin, M_out C_end );

    template<typename T>
    std::vector< std::vector< T > > dot(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B);

    template<typename T, class M_in, class M_out>
    void dotT( const M_in &A_begin, const M_in &A_end, const M_in &B_begin, const M_in &B_end, M_out C_begin, M_out C_end );

    template<typename T, class M_in, class M_out>
    void rowMajorDotT( const M_in &A_begin, const M_in &A_end, const M_in &B_begin, const M_in &B_end, const size_type rows, const size_type cols, M_out C_begin, M_out C_end );

    template<typename T>
    std::vector< std::vector< T > > dotT(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B);

    template<typename T, class M_in, class M_out>
    void Tdot( const M_in &A_begin, const M_in &A_end, const M_in &B_begin, const M_in &B_end, M_out C_begin, M_out C_end );

    template<typename T, class M_in, class M_out>
    void rowMajorTDot( const M_in &A_begin, const M_in &A_end, const M_in &B_begin, const M_in &B_end, const size_type rows, const size_type cols, M_out C_begin, M_out C_end );

    template<typename T>
    std::vector< std::vector< T > > Tdot(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B);

    template<typename T, class M_in, class M_out>
    void rowMajorTdotT(const M_in &A_begin, const M_in &A_end, const M_in &B_begin, const M_in &B_end, const size_type rows, const size_type cols, M_out C_begin, M_out C_end);

    template<typename T, class M_in, class M_out>
    void TdotT(const M_in &A_begin, const M_in &A_end, const M_in &B_begin, const M_in &B_end, M_out C_begin, M_out C_end);

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

    template<unsigned int rows, unsigned int cols, typename T, class M_in>
    void rowMajorTrace(const M_in &A_begin, const M_in &A_end, T &v);

    template<typename T, class M_in>
    void rowMajorTrace(const M_in &A_begin, const M_in &A_end, const size_type rows, T &v);

    template<typename T>
    int trace(const std::vector< T > &A, T &v);

    template<typename T>
    T trace(const std::vector< T > &A);

    template<typename T>
    int trace(const std::vector< std::vector< T > > &A, T &v);

    template<typename T>
    T trace(const std::vector< std::vector< T > > &A);

    template<typename T, class v_in>
    T l2norm(const v_in &v_begin, const v_in &v_end);

    template<typename T>
    double l2norm(const std::vector< T > &v);

    template<typename T>
    double l2norm(const std::vector< std::vector< T > > &A);

    template<typename T, class v_in>
    void unitVector(v_in v_begin, v_in v_end);

    template<typename T, class v_in, class v_out>
    void unitVector(const v_in &v_begin, const v_in &v_end, v_out unit_begin, v_out unit_end);

    template<typename T>
    std::vector< double > unitVector(const std::vector< T > &v);

    template<typename T, class v_in, class M_out>
    void dyadic( const v_in &a_begin, const v_in &a_end, const v_in &b_begin, const v_in &b_end, M_out A_begin, M_out A_end );

    template<typename T, class v_in, class M_out>
    void rowMajorDyadic( const v_in &a_begin, const v_in &a_end, const v_in &b_begin, const v_in &b_end, M_out A_begin, M_out A_end );

    template<typename T>
    std::vector< std::vector< T > > dyadic(const std::vector< T > &a, const std::vector< T > &b);

    template<typename T>
    int dyadic(const std::vector< T > &a, const std::vector< T > &b, std::vector< std::vector< T > > &A);

    template<class v_in>
    void eye(const size_type cols, v_in v_begin, v_in v_end);

    template<class M_in>
    void eye(M_in M_begin, M_in M_end);

    template<typename T>
    int eye(std::vector< T > &I);

    template<typename T>
    std::vector< std::vector< T > > eye(const unsigned int dim);

    template<typename T>
    int eye(const unsigned int dim, std::vector< std::vector< T > > &I);

    template<typename T, class v_in>
    T median(v_in v_begin, v_in v_end);

    template<typename T>
    T median(const std::vector< T > &x);

    template<class v_in>
    void abs(v_in v_begin, v_in v_end);

    template<typename T>
    std::vector< T > abs(const std::vector< T > &x);

    //Comparison Tools
    template<typename T>
    bool fuzzyEquals(const T &a, const T &b, double tolr=1e-6, double tola=1e-6);

    template<class v_in>
    bool vectorFuzzyEquals(const v_in &a_begin, const v_in &a_end, const v_in &b_begin, const v_in &b_end, double tolr=1e-6, double tola=1e-6);

    template<class M_in>
    bool matrixFuzzyEquals(const M_in &a_begin, const M_in &a_end, const M_in &b_begin, const M_in &b_end, double tolr=1e-6, double tola=1e-6);

    template<typename T>
    bool fuzzyEquals(const std::vector< T > &a, const std::vector< T > &b, double tolr=1e-6, double tola=1e-6);

    template<typename T>
    bool fuzzyEquals(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B,
        double tolr=1e-6, double tola=1e-6);

    template<typename T>
    bool equals(const T &a, const T &b);

    template<class v_in>
    bool vectorEquals(const v_in &a_begin, const v_in &a_end, const v_in &b_begin, const v_in &b_end);

    template<class M_in>
    bool matrixEquals(const M_in &a_begin, const M_in &a_end, const M_in &b_begin, const M_in &b_end);

    template<typename T>
    bool equals(const std::vector< T > &a, const std::vector< T > &b);

    template<typename T>
    bool equals(const std::vector< std::vector< T > > &a, const std::vector< std::vector< T > > &b);

    template<typename T, class v_in>
    bool isParallel( v_in v1_begin, v_in v1_end, v_in v2_begin, v_in v2_end );

    template<typename T, typename U=double>
    bool isParallel( const std::vector< T > &v1, const std::vector< T > &v2 );

    template<typename T, class v_in>
    bool isOrthogonal( v_in v1_begin, v_in v1_end, v_in v2_begin, v_in v2_end );

    template<typename T, typename U=double>
    bool isOrthogonal( const std::vector< T > &v1, const std::vector< T > &v2 );

    template<typename T>
    void verifyOrthogonal( const std::vector< T > &v1, const std::vector< T > &v2,
                           std::string message = "Vectors are not orthogonal" );

    template<class v_in>
    bool iteratorVerifyLength( const v_in &v_begin, const v_in &v_end, const unsigned int &expectedLength );

    template<class v_in>
    bool verifyLength( const v_in &v1_begin, const v_in &v1_end, const v_in &v2_begin, const v_in &v2_end );

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

    template<class v_in, class i_in, class v_out>
    void getValuesByIndex( const v_in &v_begin, const v_in &v_end, const i_in &indices_begin, const i_in &indices_end,
                           v_out subv_begin, v_out subv_end );

    template <typename T>
    int getValuesByIndex(const std::vector< T > &v, const std::vector< size_type > &indices,
        std::vector< T > &subv);

    template<class v_in, class v_out>
    void getRow( const v_in &A_begin, const v_in &A_end, const unsigned int cols, const unsigned int row, v_out row_begin );

    template <typename T>
    std::vector< T > getRow( const std::vector< T > &A, const unsigned int rows, const unsigned int cols, const unsigned int row );

    template<class v_in, class v_out>
    void getCol( const v_in &A_begin, const v_in &A_end, const unsigned int col, v_out col_begin, v_out col_end );

    template <typename T>
    std::vector< T > getCol( const std::vector< T > &A, const unsigned int rows, const unsigned int cols, const unsigned int col );

    //Appending utilities

    template<class M_in, class v_out>
    void appendVectors( const M_in &M_begin, const M_in &M_end, v_out v_begin, v_out v_end );

    template<typename T>
    std::vector< T > appendVectors(const std::vector< std::vector< T > > &A);

    template<typename T>
    std::vector< T > appendVectors(const std::initializer_list< std::vector< T > > &list);

    template< class v_in, class M_out >
    void inflate( const v_in &v_begin, const v_in &v_end, M_out M_begin, M_out M_end );

    template< typename T >
    std::vector< std::vector< T > > inflate( const std::vector< T > &Avec, const unsigned int &nrows, const unsigned int &ncols );

    //Sorting utilities
    template<class v_in, class v_out>
    void argsort( const v_in &v_begin, const v_in &v_end, v_out r_begin, v_out r_end );

    template <typename T>
    std::vector< size_type > argsort(const std::vector< T > &v);

    //Printing Utilities
    template<class v_in>
    void print(const v_in &v_begin, const v_in &v_end);

    template<class M_in>
    void printMatrix(const M_in &M_begin, const M_in &M_end);

    template<typename T>
    int print(const std::vector< T > &v);

    template<typename T>
    int print(const std::vector< std::vector< T > > &A);

    template< class v_in, class v_out >
    void rotationMatrix( const v_in &bungeEulerAngles_begin, const v_in &bungeEulerAngles_end,
                         v_out directionCosines_begin,       v_out directionCosines_end );

    template< class v_in, class v_out >
    void rotationMatrix( const v_in &bungeEulerAngles_begin,  const v_in &bungeEulerAngles_end,
                         v_out directionCosines_begin,        v_out directionCosines_end,
                         v_out dDirectionCosinesdAlpha_begin, v_out dDirectionCosinesdAlpha_end,
                         v_out dDirectionCosinesdBeta_begin,  v_out dDirectionCosinesdBeta_end,
                         v_out dDirectionCosinesdGamma_begin, v_out dDirectionCosinesdGamma_end );

    template<typename T>
    int rotationMatrix( const std::vector< T > &bungeEulerAngles, std::vector< T > &rotationMatrix );

    template<typename T>
    int rotationMatrix( const std::vector< T > &bungeEulerAngles, std::vector < std::vector< T > > &directionCosines );

    template<typename T>
    int rotationMatrix( const std::vector< T > &bungeEulerAngles, std::vector < T > &directionCosines,
                        std::vector< T > &dDirectionCosinesdAlpha,
                        std::vector< T > &dDirectionCosinesdBeta,
                        std::vector< T > &dDirectionCosinesdGamma );

    template<typename T>
    int rotationMatrix( const std::vector< T > &bungeEulerAngles, std::vector < std::vector< T > > &directionCosines,
                        std::vector< std::vector< T > > &dDirectionCosinesdAlpha,
                        std::vector< std::vector< T > > &dDirectionCosinesdBeta,
                        std::vector< std::vector< T > > &dDirectionCosinesdGamma );

    template<class v_in, class v_out, typename T=double>
    void computeMatrixExponential( const v_in &A_begin, const v_in &A_end, const size_type &dim, v_out X_begin, v_out X_end,
                                   v_out Xn_begin, v_out Xn_end, v_out expA_begin, v_out expA_end,
                                   const unsigned int nmax=40, double tola=1e-9, double tolr=1e-9 );

    template<class v_in, class v_out, class M_out, typename T=double>
    void computeMatrixExponential( const v_in &A_begin, const v_in &A_end, const size_type &dim, v_out X_begin, v_out X_end,
                                   v_out Xn_begin, v_out Xn_end,
                                   M_out dXdA_begin, M_out dXdA_end, M_out dXndA_begin, M_out dXndA_end,
                                   v_out expA_begin, v_out expA_end, M_out dExpAdA_begin, M_out dExpAdA_end,
                                   const unsigned int nmaxi=40, double tola=1e-9, double tolr=1e-9 );

    template<typename T>
    void computeMatrixExponential( const std::vector< T > &A, const unsigned int &dim, std::vector< T > &expA, const unsigned int nmax=40, double tola=1e-9, double tolr=1e-9);

    template<typename T>
    void computeMatrixExponential( const std::vector< T > &A, const unsigned int &dim, std::vector< T > &expA, std::vector< T > & dExpAdA, const unsigned int nmax=40, double tola=1e-9, double tolr=1e-9);

    template<class v_in, class v_out, typename T=double>
    void computeMatrixExponentialScalingAndSquaring( const v_in &A_begin, const v_in &A_end, const size_type &dim,
                                                     v_out tempVector1_begin, v_out tempVector1_end,
                                                     v_out tempVector2_begin, v_out tempVector2_end,
                                                     v_out tempVector3_begin, v_out tempVector3_end,
                                                     v_out expA_begin, v_out expA_end,
                                                     const unsigned int nmax=40, double tola=1e-9, double tolr=1e-9 );

    template<class v_in, class v_out, class M_out, typename T=double>
    void computeMatrixExponentialScalingAndSquaring( const v_in &A_begin, const v_in &A_end, const size_type &dim,
                                                     v_out tempVector1_begin, v_out tempVector1_end,
                                                     v_out tempVector2_begin, v_out tempVector2_end,
                                                     v_out tempVector3_begin, v_out tempVector3_end,
                                                     M_out tempMatrix1_begin, M_out tempMatrix1_end,
                                                     M_out tempMatrix2_begin, M_out tempMatrix2_end,
                                                     v_out expA_begin, v_out expA_end, M_out dExpAdA_begin, M_out dExpAdA_end,
                                                     const unsigned int nmaxi=40, double tola=1e-9, double tolr=1e-9 );

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

        template<class M_in, class v_in, class v_out, typename T, int R=-1, int C=-1>
        void solveLinearSystem( const M_in &A_begin, const M_in &A_end, const v_in &b_begin, const v_in &b_end,
                                const unsigned int nrows, const unsigned int ncols, v_out x_begin, v_out x_end,
                                unsigned int &rank, solverType< T > &linearSolver );

        template<class v_in, typename T, int R=-1, int C=-1>
        T determinant( const v_in &A_begin, const v_in &A_end, const unsigned int nrows, const unsigned int ncols );

        template<typename T>
        T determinant(const std::vector< T > &Avec, const unsigned int nrows, const unsigned int ncols);

        template<typename T, class M_in, class M_out, int R=-1, int C=-1>
        void inverse( const M_in &A_begin, const M_in &A_end, const unsigned int nrows, const unsigned int ncols,
                      M_out Ainv_begin,    M_out Ainv_end );

        template<typename T>
        std::vector< double > inverse(const std::vector< T > &Avec, const unsigned int nrows, const unsigned int ncols);

        template<typename T>
        std::vector< std::vector< double > > inverse( const std::vector< std::vector< T > > &A );

        template<class M_in, class M_out>
        void computeFlatDInvADA( const M_in &invA_begin, const M_in &invA_end, const unsigned int nrows, const unsigned int ncols,
                                 M_out result_begin, M_out result_end );

        template<int nrows, int ncols, class M_in, class M_out>
        void computeFlatDInvADA( const M_in &invA_begin, const M_in &invA_end, M_out result_begin, M_out result_end );

        template<typename T>
        std::vector< double > computeFlatDInvADA( const std::vector< T > &invA, const unsigned int nrows, const unsigned int ncols );

        template<typename T>
        std::vector< std::vector< double > > computeDInvADA( const std::vector< T > &invA, const unsigned int nrows, const unsigned int ncols );

        template<class M_in, class M_out, typename T, int R=-1, int C=-1>
        void computeDDetADA(const M_in &A_begin, const M_in &A_end, const unsigned int nrows, const unsigned int ncols,
                            M_out result_begin, const M_out result_end );

        template<typename T>
        std::vector< T > computeDDetADA(const std::vector< T > &Avec, const unsigned int nrows, const unsigned int ncols);

        template< typename T >
        std::vector< T > matrixMultiply(const std::vector< T > &A, const std::vector< T > &B,
                                             const unsigned int Arows, const unsigned int Acols,
                                             const unsigned int Brows, const unsigned int Bcols,
                                             const bool Atranspose = false, const bool Btranspose = false);

        template< class v_in, class v_out, class M_out >
        int __matrixSqrtResidual( const v_in &A_begin, const v_in &A_end,
                                  const unsigned int Arows,
                                  v_out X_begin, v_out X_end,
                                  v_out R_begin, v_out R_end,
                                  M_out J_begin, M_out J_end );

        template< typename T >
        std::vector< double > matrixSqrt(const std::vector< T > &A, const unsigned int Arows,
                                         const double tolr = 1e-9, const double tola = 1e-9, const unsigned int maxIter = 20,
                                         const unsigned int maxLS = 5);

        template< typename T >
        std::vector< double > matrixSqrt(const std::vector< T > &A, const unsigned int Arows,
                                         std::vector< double > &dSqrtAdX,
                                         const double tolr = 1e-9, const double tola = 1e-9, const unsigned int maxIter = 20,
                                         const unsigned int maxLS = 5);

        template< typename T >
        std::vector< double > matrixSqrt(const std::vector< T > &A, const unsigned int Arows,
                                         std::vector< std::vector< double > > &dSqrtAdX,
                                         const double tolr = 1e-9, const double tola = 1e-9, const unsigned int maxIter = 20,
                                         const unsigned int maxLS = 5);

        template< typename T, class v_in, class v_out, class M_out >
        int matrixSqrt( const v_in A_begin, const v_in A_end, const unsigned int Arows,
                        v_out X_begin, v_out X_end, v_out dX_begin, v_out dX_end,
                        v_out R_begin, v_out R_end, M_out dSqrtAdX_begin, M_out dSqrtAdX_end,
                        const double tolr=1e-9, const double tola=1e-9, const unsigned int maxIter=20,
                        const unsigned int maxLS=5 );

        template<typename T, class M_in, class M_out, class v_out, int R=-1, int C=-1 >
        void svd( const M_in &A_begin, const M_in &A_end, const unsigned int nrows, const unsigned int ncols,
                  M_out U_begin, M_out U_end, v_out Sigma_begin, v_out Sigma_end, M_out V_begin, M_out V_end );

        template< typename T >
        void svd( const std::vector< T > &A, std::vector< std::vector< double > > &U, std::vector< double > &Sigma,
                  std::vector< std::vector< double > > &V );

        template< typename T, class v_in, class v_out, class M_out, int R=-1, int C=-1 >
        void polar_decomposition( const v_in &A_begin, const v_in &A_end, const unsigned int nrows, const unsigned int ncols,
                                  v_out Usqrd_begin, v_out Usqrd_end, v_out tempVec1_begin, v_out tempVec1_end, v_out tempVec2_begin, v_out tempVec2_end,
                                  M_out dUdUsqrd_begin, M_out dUdUsqrd_end, v_out R_begin, v_out R_end, v_out U_begin, v_out U_end, const bool left );

        template< typename T >
        void polar_decomposition( const std::vector< T > &A, const unsigned int nrows, const unsigned int ncols,
                                  std::vector< double > &R, std::vector< double > &U, const bool left = false );

    #endif

}

#include "tardigrade_vector_tools.cpp"
#endif
