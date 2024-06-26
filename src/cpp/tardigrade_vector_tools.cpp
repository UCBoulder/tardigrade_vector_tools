/**
  *****************************************************************************
  * \file tardigrade_vector_tools.cpp
  *****************************************************************************
  * A collection of functions and related utilities intended to help perform
  * vector operations in cpp.
  *****************************************************************************
  */

#include "tardigrade_vector_tools.h"

//Operator overloading
template<typename T>
std::vector<T>& operator+=(std::vector<T> &lhs, const std::vector<T> &rhs){
    /*!
     * Overload the += operator for vectors
     *
     * \param &lhs: The left-hand side vector
     * \param &rhs: The right-hand side vector
     */

    const unsigned int lhs_size = lhs.size( );
    const unsigned int rhs_size = rhs.size( );

    TARDIGRADE_ERROR_TOOLS_CHECK( lhs_size == rhs_size, "vectors must be the same size to add" )

    std::transform(lhs.begin( ), lhs.end( ),
                   rhs.begin( ), lhs.begin( ),
                   [](T i, T j){return i+j;});

    return lhs;
}

template<typename T>
std::vector<T>& operator+=(std::vector<T> &lhs, const T &rhs){
    /*!
     * Overload the += operator for vector scalar addition
     *
     * \param &lhs: The left-hand side vector
     * \param &rhs: The scalar being added to the vector
     */

    std::transform(lhs.begin( ), lhs.end( ), lhs.begin( ),
                   std::bind(std::plus<T>(), std::placeholders::_1, rhs ));
    return lhs;
}

template<typename T>
std::vector<T> operator+(std::vector<T> lhs, const std::vector<T> &rhs){
    /*!
     * Overload the + operator for vectors
     *
     * \param &lhs: The left-hand side vector
     * \param &rhs: The right-hand side vector
     */

    return lhs += rhs;
}

template<typename T>
std::vector<T> operator+(std::vector<T> lhs, const T &rhs){
    /*!
     * Overload the + operator for vector - scalar addition
     *
     * \param &lhs: The left-hand side vector
     * \param &rhs: The right-hand side vector
     */

    return lhs += rhs;
}

template<typename T>
std::vector<T> operator+(const T &lhs, std::vector<T> rhs){
    /*!
     * Overload the + operator for vectors
     *
     * \param &lhs: The left-hand side vector
     * \param &rhs: The right-hand side vector
     */

    return rhs += lhs;
}

template<typename T>
std::vector<T> operator-(std::vector<T> v){
    /*!
     * Overload the negative operator for vectors
     *
     * \param &v: The vector in question
     */

    std::transform(v.cbegin( ), v.cend( ), v.begin( ), std::negate<T>( ));

    return v;
}

template<typename T>
std::vector<T>& operator-=(std::vector<T> &lhs, const std::vector<T> &rhs){
    /*!
     * Overload the -= operator for vectors
     *
     * \param &lhs: The left-hand side vector
     * \param &rhs: The right-hand side vector
     */
    return lhs += -rhs;
}

template<typename T>
std::vector<T>& operator-=(std::vector<T> &lhs, const T &rhs){
    /*!
     * Overload the subtraction operator for vector - scalar pairs
     *
     * \param &lhs: The left-hand side vector.
     * \param &rhs: The right-hand side scalar.
     */
    return lhs += -rhs;
}

template<typename T>
std::vector<T> operator-(std::vector<T> lhs, const std::vector<T> &rhs){
    /*!
     * Overload the subtraction operator for vectors
     *
     * \param &lhs: The left-hand side vector
     * \param &rhs: The right-hand side vector
     */

    return lhs -= rhs;
}

template<typename T>
std::vector<T> operator-(std::vector<T> lhs, const T &rhs){
    /*!
     * Overload the subtraction operator for vector - scalar pairs
     *
     * \param lhs: The left-hand side vector
     * \param &rhs: The right-hand side scalar
     */

    return lhs -= rhs;
}

template<typename T>
std::vector<T> operator-(const T &lhs, std::vector<T> rhs){
    /*!
     * Overload the subtraction operator for vector - scalar pairs
     *
     * \param &lhs: The left-hand side scalar
     * \param rhs: The right-hand side vector
     */

    rhs -= lhs;
    return -rhs;
}

template<typename T, typename t>
std::vector<T>& operator*=(std::vector<T> &lhs, const t rhs){
    /*!
     * Overload the *= operator for vectors
     *
     * \param lhs: The left-hand side vector
     * \param rhs: The right-hand side scalar
     */
    std::transform( lhs.begin( ), lhs.end( ), lhs.begin( ),
                    std::bind( std::multiplies<T>(), std::placeholders::_1, rhs));

    return lhs;
}

template<typename T, typename t>
std::vector<T> operator*(const t lhs, std::vector<T> rhs){
    /*!
     * Overload the * operator for vectors
     *
     * \param lhs: The left-hand side scalar
     * \param rhs: The right-hand side vector
     */
    return rhs*=lhs;
}


template<typename T, typename t>
std::vector<T> operator*(std::vector<T> lhs, const t rhs){
    /*!
     * Overload the * operator for vectors
     *
     * \param lhs: The left-hand side vector
     * \param rhs: The right-hand side scalar
     */
    return lhs*=rhs;
}

template<typename T, typename t>
std::vector<T>& operator/=(std::vector<T> &lhs, const t rhs){
    /*!
     * Overload the /= operator for vectors
     *
     * \param lhs: The left-hand side vector
     * \param rhs: The right-hand side scalar
     */
    return lhs*=(1./rhs);
}

template<typename T, typename t>
std::vector<T> operator/(std::vector<T> lhs, const t rhs){
    /*!
     * Overload the / operator for vectors
     *
     * \param lhs: The left-hand side vector
     * \param rhs: The right-hand side scalar
     */
    return lhs/=rhs;
}

template<typename T>
std::vector< std::vector< T > >& operator+=(std::vector< std::vector< T > > &lhs, const std::vector< std::vector< T > > &rhs){
    /*!
     * Overload the += operator for matrices
     *
     * \param lhs: The left-hand side matrix
     * \param &rhs: The right-hand side matrix
     */

    const unsigned int lhs_size = lhs.size( );
    const unsigned int rhs_size = rhs.size( );

    TARDIGRADE_ERROR_TOOLS_CHECK( lhs_size == rhs_size, "matrices must have the same numbers of rows to add")

    std::transform( lhs.begin( ), lhs.end( ), rhs.begin( ), lhs.begin( ),
                    [](std::vector<T> a, std::vector<T> b){ return a += b; } );

    return lhs;
}

template<typename T>
std::vector< std::vector< T > > operator+(std::vector< std::vector< T > > lhs, const std::vector< std::vector< T > > &rhs){
    /*!
     * Overload the + operator for matrices
     *
     * \param lhs: The left-hand side matrix
     * \param &rhs: The right-hand side matrix
     */

    return lhs += rhs;
}

template<typename T>
std::vector< std::vector< T > > operator-(std::vector< std::vector< T > > v){
    /*!
     * Overload the negation operator for matrices.
     *
     * \param v: The matrix to negate.
     */

    std::transform(v.begin( ), v.end( ), v.begin( ), [](std::vector<T> vi){return -vi;});
    return v;
}

template<typename T>
std::vector< std::vector < T > >& operator-=(std::vector< std::vector< T > > &lhs, const std::vector< std::vector< T > > &rhs){
    /*!
     * Overload the -= operator for matrices
     *
     * \param lhs: The left-hand side matrix
     * \param &rhs: The right-hand side matrix
     */

    const unsigned int lhs_size = lhs.size( );
    const unsigned int rhs_size = rhs.size( );

    TARDIGRADE_ERROR_TOOLS_CHECK( lhs_size == rhs_size, "matrices must have the same numbers of rows to add")

    for (unsigned int i=0; i<lhs_size; i++){
        lhs[i] += -rhs[i];
    }

    return lhs;
}

template<typename T>
std::vector< std::vector< T > > operator-(std::vector< std::vector< T > > lhs, const std::vector< std::vector< T > > &rhs){
    /*!
     * Overload the - operator for matrices
     *
     * \param lhs: The left-hand side matrix
     * \param &rhs: The right-hand side matrix
     */
    return lhs -= rhs;
}

template<typename T, typename t>
std::vector<std::vector<T>>& operator*=(std::vector<std::vector<T>> &lhs, const t rhs){
    /*!
     * Overload the *= operator for matrices
     *
     * \param lhs: The left-hand side matrix
     * \param rhs: The right-hand side scalar
     */
    for ( auto li=lhs.begin(); li!=lhs.end(); li++ ){
        *li *= rhs;
    }

    return lhs;
}

template<typename T, typename t>
std::vector<std::vector<T>> operator*(const t lhs, std::vector<std::vector<T>> rhs){
    /*!
     * Overload the * operator for matrices
     *
     * \param lhs: The left-hand side scalar
     * \param rhs: The right-hand side matrix
     */
    return rhs*=lhs;
}


template<typename T, typename t>
std::vector<std::vector<T>> operator*(std::vector<std::vector<T>> lhs, const t rhs){
    /*!
     * Overload the * operator for matrices
     *
     * \param lhs: The left-hand side matrix
     * \param rhs: The right-hand side scalar
     */
    return lhs*=rhs;
}

template<typename T, typename t>
std::vector<std::vector<T>>& operator/=(std::vector<std::vector<T>> &lhs, const t rhs){
    /*!
     * Overload the /= operator for matrices
     *
     * \param lhs: The left-hand side matrix
     * \param rhs: The right-hand side scalar
     */
    return lhs*=(1./rhs);
}

template<typename T, typename t>
std::vector<std::vector<T>> operator/(std::vector<std::vector<T>> lhs, const t rhs){
    /*!
     * Overload the / operator for matrices
     *
     * \param lhs: The left-hand side matrices
     * \param rhs: The right-hand side scalar
     */
    return lhs/=rhs;
}

template<typename T>
std::vector<std::vector<T>>& operator+=(std::vector<std::vector<T>> &lhs, const T &rhs){
    /*!
     * Overload the += operator for matrix scalar addition
     *
     * \param &lhs: The left-hand side matrix
     * \param &rhs: The scalar being added to the matrix
     */

    for ( auto li = lhs.begin( ); li != lhs.end( ); li++ ){
        *li += rhs;
    }

    return lhs;
}

template<typename T>
std::vector<std::vector<T>> operator+(std::vector<std::vector<T>> lhs, const T &rhs){
    /*!
     * Overload the + operator for matrix - scalar addition
     *
     * \param &lhs: The left-hand side matrix
     * \param &rhs: The right-hand side scalar
     */

    return lhs += rhs;
}

template<typename T>
std::vector<std::vector<T>> operator+(const T &lhs, std::vector<std::vector<T>> rhs){
    /*!
     * Overload the + operator for matrix - scalar addition
     *
     * \param &lhs: The left-hand side scalar
     * \param &rhs: The right-hand side matrix
     */

    return rhs += lhs;
}

namespace tardigradeVectorTools{

    //Computation Utilities
    template<typename T>
    int computeMean(const std::vector< std::vector< T > > &A, std::vector< T > &v){
        /*!
         * Compute the column-wise mean of A
         *
         * \param &A: The matrix of vectors
         * \param &v: The resulting mean
         */

        const unsigned int A_size = A.size( );

        TARDIGRADE_ERROR_TOOLS_CHECK( A_size != 0, "Matrix must have a size greater than zero" );

        //Size the output vector
        v = std::vector<T>(A[0].size(), 0);

        for ( auto Ai = A.begin( ); Ai != A.end( ); Ai++ ){
            v += *Ai;
        }

        v /= A_size;

        return 0;
    }

    template<typename T>
    std::vector< T > computeMean(const std::vector< std::vector< T > > &A){
        /*!
         * Compute the column-wise mean of A
         *
         * \param &A: The matrix of vectors
         */

        std::vector< T > v;
        computeMean(A, v);
        return v;
    }

    template<typename T>
    int cross(const std::vector< T > &a, const std::vector< T > &b, std::vector< T > &c){
        /*!
         * Compute the cross product of two vectors i.e. a x b
         * Note that if a and b are 2D vectors a 3D vector for c will be returned.
         *
         * TODO: Generalize this to n dimensions.
         *
         * \param &a: The first vector
         * \param &b: The second vector
         * \param &c: The resulting vector
         */

        size_type size = a.size();
        c = std::vector< T >(size, 0);

        if (size == 2){
            c.resize(3);
            c[2] =  a[0]*b[1] - a[1]*b[0];
        }
        else if (size == 3){
            c[0] =  a[1]*b[2] - a[2]*b[1];
            c[1] = -a[0]*b[2] + a[2]*b[0];
            c[2] =  a[0]*b[1] - a[1]*b[0];
        }
        else{
            TARDIGRADE_ERROR_TOOLS_CHECK( false, "Only 2D and 3D vectors are accepted");
        }

        return 0;
    }

    template<typename T>
    std::vector< T > cross(const std::vector< T > &a, const std::vector< T > &b){
        /*!
         * Compute the cross product of two vectors i.e. a x b
         * Note that if a and b are 2D vectors a 3D vector for c will be returned
         *
         * TODO: Generalize this to n dimensions
         *
         * \param &a: The first vector
         * \param &b: The second vector
         */

         std::vector< T > c;
         cross(a, b, c);
         return c;
    }

    template<typename T>
    int dot(const std::vector< T > &a, const std::vector< T > &b, T &v){
        /*!
         * Compute the dot product of two vectors i.e. v = a_i b_i
         *
         * \param &a: The first vector
         * \param &b: The second vector
         * \param &v: The output quantity
         */

        //Get the size and perform error handling
        size_type size = a.size();
        TARDIGRADE_ERROR_TOOLS_CHECK( size == b.size(), "vectors must be the same size to compute the dot product" )

        //Set v to zero
        v = std::inner_product(a.begin(), a.end(), b.begin(), T());

        return 0;
    }

    template<typename T>
    T dot(const std::vector< T > &a, const std::vector< T > &b){
        /*!
         * Compute the dot product of two vectors i.e. v = a_i b_i
         *
         * \param &a: The first vector
         * \param &b: The second vector
         */

        T v;
        dot(a, b, v);
        return v;
    }

    template<typename T>
    std::vector< T > dot(const std::vector< std::vector< T > > &A, const std::vector< T > &b){
        /*!
         * Compute the dot product between a matrix and a vector resulting i.e. c_i = A_ij b_j
         *
         * \param &A: The matrix
         * \param &b: The vector
         */

        size_type size = A.size();

        std::vector< T > c(size);

        unsigned int i=0;
        for (auto A_i=A.begin(); A_i!=A.end(); A_i++, i++){
            c[i] = dot(*A_i, b);
        }
        return c;
    }

    template<typename T>
    std::vector< T > Tdot(const std::vector< std::vector< T > > &A, const std::vector< T > &b){
        /*!
         * Compute the dot product between a matrix and a vector resulting i.e. c_i = A_ji b_j
         *
         * \param &A: The matrix
         * \param &b: The vector
         */

        size_type size = A.size();

        TARDIGRADE_ERROR_TOOLS_CHECK( size != 0, "A has no rows")

        TARDIGRADE_ERROR_TOOLS_CHECK( size == b.size(), "A and b are incompatible shapes");

        std::vector< T > c(A[0].size(), 0);

        for ( unsigned int i = 0; i < A[0].size(); i++ ){
            for ( unsigned int j = 0; j < size; j++ ){
                c[i] += A[j][i] * b[j];
            }
        }
        return c;
    }

    template<typename T>
    std::vector< std::vector< T > > dot(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B){
        /*!
         * Compute the dot product between two matrices i.e. C_{ij} = A_{ik} B_{kj}
         *
         * \param &A: The first matrix
         * \param &B: The second matrix
         */

        size_type rows = A.size();

        TARDIGRADE_ERROR_TOOLS_CHECK( B.size() != 0, "B has no rows")

        size_type inner = B.size();
        size_type cols = B[0].size();

        //Perform the matrix multiplication
        std::vector< std::vector< T > > C(rows, std::vector< T >(cols, 0));

        for (unsigned int I=0; I<rows; I++){

            TARDIGRADE_ERROR_TOOLS_CHECK( A[I].size( ) == inner, "A and B have incompatible shapes" )

            for (unsigned int K=0; K<inner; K++){

                TARDIGRADE_ERROR_TOOLS_CHECK( B[K].size() == cols, "B is not a regular matrix" )

                for (unsigned int J=0; J<cols; J++){

                    C[I][J] += A[I][K] * B[K][J];
                }
            }
        }
        return C;
    }

    template<typename T>
    std::vector< std::vector< T > > dotT(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B){
        /*!
         * Compute the dot product between two matrices where the second is transposed i.e. C_{ij} = A_{ik} B_{jk}
         *
         * \param &A: The first matrix
         * \param &B: The second matrix
         */

        size_type Arows = A.size();

        TARDIGRADE_ERROR_TOOLS_CHECK( B.size( ) != 0, "B has no rows" );

        size_type Brows = B.size();
        size_type Bcols = B[0].size();

        //Perform the matrix multiplication
        std::vector< std::vector< T > > C(Arows, std::vector< T >(Brows, 0));

        for (unsigned int I=0; I<Arows; I++){

            TARDIGRADE_ERROR_TOOLS_CHECK( A[I].size() == Bcols, "A and B have incompatible shapes" );

            for (unsigned int J=0; J<Brows; J++){

                TARDIGRADE_ERROR_TOOLS_CHECK( B[J].size() == Bcols, "B is not a regular matrix" );

                for (unsigned int K=0; K<Bcols; K++){

                    C[I][J] += A[I][K] * B[J][K];
                }
            }
        }
        return C;
    }

    template<typename T>
    std::vector< std::vector< T > > Tdot(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B){
        /*!
         * Compute the dot product between two matrices where the first is transposed i.e. C_{ij} = A_{ki} B_{kj}
         *
         * \param &A: The first matrix
         * \param &B: The second matrix
         */

        size_type Arows = A.size();

        TARDIGRADE_ERROR_TOOLS_CHECK( Arows != 0, "A has no rows" )

        size_type Acols = A[0].size();

        TARDIGRADE_ERROR_TOOLS_CHECK( B.size( ) != 0, "B has no rows" );

        size_type Brows = B.size();
        size_type Bcols = B[0].size();

        TARDIGRADE_ERROR_TOOLS_CHECK( Arows == Brows, "A and B have incompatible shapes" );

        //Perform the matrix multiplication
        std::vector< std::vector< T > > C(Acols, std::vector< T >(Bcols, 0));

        for (unsigned int I=0; I<Acols; I++){

            for (unsigned int J=0; J<Bcols; J++){

                for (unsigned int K=0; K<Brows; K++){

                    TARDIGRADE_ERROR_TOOLS_CHECK( B[K].size() == Bcols, "B is not a regular matrix" )

                    TARDIGRADE_ERROR_TOOLS_CHECK( A[K].size() == Acols, "A is not a regular matrix" )

                    C[I][J] += A[K][I] * B[K][J];
                }
            }
        }
        return C;
    }

    template<typename T>
    std::vector< std::vector< T > > TdotT(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B){
        /*!
         * Compute the dot product between two matrices where both are transposed i.e. C_{ij} = A_{ki} B_{jk}
         *
         * \param &A: The first matrix
         * \param &B: The second matrix
         */

        std::vector< std::vector< T > > CT = dot(B, A);

        std::vector< std::vector< T > > C(CT[0].size(), std::vector< T > (CT.size()));

        for (unsigned int i=0; i<C.size(); i++){
            for (unsigned int j=0; j<C[i].size(); j++){
                C[i][j] = CT[j][i];
            }
        }
        return C;
    }

    template<typename T>
    int inner(const std::vector< T > &A, const std::vector< T > &B, T &result){
        /*!
         * Compute the inner product between two matrices stored in row major format
         *
         * \f$result = \sum{A_{ij}*B_{ij}}\f$
         *
         * \param &A: The first matrix in row major format
         * \param &B: The second matrix in row major format
         * \param &result: The inner product scalar
         */

        result = 0.;
        dot(A, B, result);

        return 0;
    }

    template<typename T>
    T inner(const std::vector< T > &A, const std::vector< T > &B){
        /*!
         * Compute the inner product between two matrices stored in row major format
         *
         * \f$result = \sum{A_{ij}*B_{ij}}\f$
         *
         * \param &A: The first matrix in row major format
         * \param &B: The second matrix in row major format
         * :return: The inner product scalar
         * :rtype: T result
         */

        T result = 0.;
        inner(A, B, result);

        return result;
    }

    template<typename T>
    int inner(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B, T &result){
        /*!
         * Compute the inner product between two matrices stored in matrix format
         *
         * \f$result = \sum{A_{ij}*B_{ij}}\f$
         *
         * \param &A: The first matrix
         * \param &B: The second matrix
         * \param &result: The inner product scalar
         */

        //Get the size and perform error handling
        unsigned int Arows = A.size();
        unsigned int Acols = A[0].size();
        TARDIGRADE_ERROR_TOOLS_CHECK( Arows == B.size() && Acols == B[0].size(), "Matrices must have the same dimensions to add.")

        //Convert to row major matrices
        std::vector< T > Avec = appendVectors(A);
        std::vector< T > Bvec = appendVectors(B);

        result = 0.;
        inner(Avec, Bvec, result);

        return 0;
    }

    template<typename T>
    T inner(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B){
        /*!
         * Compute the inner product between two matrices stored in matrix format
         *
         * \f$result = \sum{A_{ij}*B_{ij}}\f$
         *
         * \param &A: The first matrix
         * \param &B: The second matrix
         * :return: The inner product scalar
         * :rtype: T result
         */

        T result = 0.;
        inner(A, B, result);

        return result;
    }


    template<typename T>
    int trace(const std::vector< T > &A, T &v){
        /*!
         * Compute the trace of a square matrix ( \f$A\f$ ) in row major format:
         *
         * \f$v = A_{ii}\f$
         *
         * \param &A: The matrix in row major format ( \f$A\f$ )
         * \param &v: The scalar output quantity ( \f$v\f$ )
         */

        //Get the size and perform error handling
        unsigned int length = A.size();
        unsigned int dimension = std::round(std::sqrt(length));
        TARDIGRADE_ERROR_TOOLS_CHECK( dimension*dimension == length, "The trace can only be computed for square matrices.")

        //Set v to zero
        v = 0;

        //Compute the trace
        for (size_type i=0; i<dimension; i++){
            v += A[dimension*i + i];
        }

        return 0;
    }

    template<typename T>
    T trace(const std::vector< T > &A){
        /*!
         * Compute the trace of a square matrix in row major format
         *
         * \f$v = A_{ii}\f$
         *
         * \param &A: The matrix in row major format ( \f$A\f$ )
         */

        T v;
        trace(A, v);
        return v;
    }

    template<typename T>
    int trace(const std::vector< std::vector< T > > &A, T &v){
        /*!
         * Compute the trace of a square matrix
         *
         * \f$v = A_{ii}\f$
         *
         * \param &A: The matrix
         * \param &v: The scalar output quantity
         */

        //Convert matrix to row major vector format
        std::vector< T > Avec = appendVectors(A);

        trace(Avec, v);
        return 0;
    }

    template<typename T>
    T trace(const std::vector< std::vector< T > > &A){
        /*!
         * Compute the trace of a square matrix
         *
         * \f$v = A_{ii}\f$
         *
         * \param &A: The matrix
         */

        T v;
        trace(A, v);
        return v;
    }

    template<typename T>
    double l2norm(const std::vector< T > &v){
        /*!
         * Compute the l2 norm of the vector v i.e. (v_i v_i)^(0.5)
         *
         * \param &v: The vector to compute the norm of
         */

        return std::sqrt(dot(v, v));
    }

    template<typename T>
    double l2norm(const std::vector< std::vector < T > > &A){
        /*!
         * Compute the l2 norm of the matrix A i.e. (A_ij A_ij)^(0.5)
         *
         * \param &A: The matrix to compute the norm of
         */

        double v=0;
        for (auto it=A.begin(); it!=A.end(); it++){
            v += dot(*it, *it);
        }
        return std::sqrt(v);
    }

    template<typename T>
    std::vector< double > unitVector(const std::vector< T > &v){
        /*!
         * Compute the unit vector v i.e. \f$v_j / (v_i v_i)^(0.5)\f$
         *
         * \param &v: The vector to compute the norm of
         */
        //Recast the incoming vectors as double
        std::vector< double > vDouble(v.begin(), v.end());
        return vDouble / l2norm(vDouble);
    }

    template<typename T>
    std::vector< std::vector< T > > dyadic(const std::vector< T > &a, const std::vector< T > &b){
        /*!
         * Compute the dyadic product between two vectors returning a matrix i.e. A_ij = a_i b_j;
         */

        std::vector< std::vector< T > > A;
        dyadic(a, b, A);
        return A;
    }

    template<typename T>
    int dyadic(const std::vector< T > &a, const std::vector< T > &b, std::vector< std::vector< T > > &A){
        /*!
         * Compute the dyadic product between two vectors return a matrix i.e. A_ij = a_i b_j
         *
         * \param &a: The first vector
         * \param &b: The second vector
         * \param &A: The returned matrix
         */

        A.resize(a.size());
        unsigned int i=0;
        for (auto ai=a.begin(); ai!=a.end(); ai++, i++){
            A[i].resize(b.size());
            unsigned int j=0;
            for (auto bj=b.begin(); bj!=b.end(); bj++, j++){
                A[i][j] = (*ai)*(*bj);
            }
        }
        return 0;
    }

    template<typename T>
    int eye(std::vector< T > &I){
        /*!
         * Construct an identity tensor in row major format
         *
         * \param &I: The identity matrix
         */

        //Get the size and perform error handling
        unsigned int length = I.size();
        unsigned int dimension = std::round(std::sqrt(length));
        TARDIGRADE_ERROR_TOOLS_CHECK( dimension*dimension == length, "The identity tensor can only be constructed for square matrices.")

        //Construct the identity matrix
        I = std::vector< T >(I.size(), 0);
        for (size_type i=0; i<dimension; i++){
            I[dimension*i + i] = 1;
        }

        return 0;
    }

    template<typename T>
    std::vector< std::vector< T > > eye(const unsigned int dim){
        /*!
         * Construct an identity tensor of the size indicated by dim
         *
         * \param dim: The dimension of the matrix
         */

        std::vector< std::vector< T > > I(dim, std::vector< T >(dim, 0));
        for (unsigned int i=0; i<dim; i++){
            I[i][i] = 1;
        }
        return I;
    }

    template<typename T>
    int eye(const unsigned int dim, std::vector< std::vector< T > > &I){
        /*!
         * Construct an identity tensor of the size indicated by dim
         *
         * \param dim: The dimension of the matrix
         * \param &I: The resulting identity matrix
         */

        I = eye<T>(dim);

        return 0;
    }

    template< typename T >
    T median(const std::vector< T > &x){
        /*!
         * Compute the median of a vector x
         *
         * \param &x: The vector to compute the median of.
         */

        unsigned int n = x.size();
        std::vector< T > xcopy = x;
        std::sort(xcopy.begin(), xcopy.end());

        if ( (n & 2) == 0){
            return xcopy[n / 2];
        }
        else{
            return 0.5*( xcopy[(n - 1)/2] + xcopy[n / 2] );
        }
    }

    template< typename T >
    std::vector< T > abs(const std::vector< T > &x){
        /*!
         * Compute the absolute value of every component of a vector.
         *
         * \param &x: The vector to compute the absolute value of.
         */

        std::vector< T > xcopy = x;
        for (unsigned int i=0; i<xcopy.size(); i++){
            xcopy[i] = std::abs(xcopy[i]);
        }
        return xcopy;
    }

    //Comparison Utilities
    template< typename T >
    bool fuzzyEquals(const T &a, const T &b, double tolr, double tola){
        /*!
         * Compare two values to determine if they are equal within a
         * tolerance.
         *
         * \param &a: The first value to compare
         * \param &b: The second value to compare
         * \param tolr: The relative tolerance
         * \param tola: The absolute tolerance
         */

        double tol = fmin(tolr*fabs(a) + tola, tolr*fabs(b) + tola);
        return fabs(a-b)<tol;
    }

    template< typename T >
    bool fuzzyEquals(const std::vector< T > &a, const std::vector< T > &b, double tolr, double tola){
        /*!
         * Compare two vectors to determine if they are equal within a
         * tolerance.
         *
         * \param &a: The first vector to compare
         * \param &b: The second vector to compare
         * \param tolr: The relative tolerance
         * \param tola: The absolute tolerance
         */

        if (a.size() != b.size()){
            return false;
        }

        for (unsigned int i=0; i<a.size(); i++){
            if (!fuzzyEquals(a[i], b[i], tolr, tola)){
                return false;
            }
        }
        return true;
    }

    template< typename T >
    bool fuzzyEquals(const std::vector< std::vector< T > > &A, const std::vector< std::vector< T > > &B, double tolr, double tola){
        /*!
         * Compare two matrices to determine if they are equal within a
         * tolerance.
         *
         * \param &A: The first matrix to compare
         * \param &B: The second matrix to compare
         * \param tolr: The relative tolerance
         * \param tola: The absolute tolerance
         */

        if (A.size() != B.size()){
            return false;
        }

        for (unsigned int i=0; i<A.size(); i++){
            if (!fuzzyEquals(A[i], B[i], tolr, tola)){
                return false;
            }
        }
        return true;
    }

    template<typename T>
    bool equals(const T &a, const T &b){
        /*!
         * Compare two values for exact equality
         * \param &a: The first value to compare
         * \param &b: The second value to compare
         */
         return a == b;
    }

    template<typename T>
    bool equals(const std::vector< T > &a, const std::vector< T > &b){
        /*!
         * Compare two vectors for exact equality
         *
         * \param &a: The first vector to compare
         * \param &b: The second vector to compare
         */
        unsigned int size = a.size();
        if (size != b.size()){
            return false;
        }
        for (unsigned int i=0; i<size; i++){
            if (!equals(a[i], b[i])){
                return false;
            }
        }
        return true;
    }

    template<typename T>
    bool equals(const std::vector< std::vector< T > > &a, const std::vector< std::vector< T > > &b){
        /*!
         * Compare two matrices for exact equality
         *
         * \param &a: The first matrix to compare
         * \param &b: The second matrix to compare
         */

        unsigned int size = a.size();
        if (size != b.size()){
            return false;
        }
        for (unsigned int i=0; i<size; i++){
            if (!equals(a[i], b[i])){
                return false;
            }
        }
        return true;
    }

    template<typename T>
    bool isParallel( const std::vector< T > &v1, const std::vector< T > &v2 ){
        /*!
         * Compare two vectors and determine if they are parallel
         *
         * \param &v1: The first vector
         * \param &v2: The second vector
         */

        //Compute the unit vector for each
        std::vector< double > nv1 = unitVector( v1 );
        std::vector< double > nv2 = unitVector( v2 );

        //Compute the distance
        double d = std::abs(dot(nv1, nv2));

        return fuzzyEquals(d, 1.);
    }

    template<typename T>
    bool isOrthogonal( const std::vector< T > &v1, const std::vector< T > &v2 ){
        /*!
         * Compare two vectors and determine if they are orthogonal
         *
         * \param &v1: The first vector
         * \param &v2: The second vector
         */

        //Compute the unit vector for each
        std::vector< double > nv1 = unitVector( v1 );
        std::vector< double > nv2 = unitVector( v2 );

        //Compute the distance
        double d = std::abs(dot(nv1, nv2));

        return fuzzyEquals(d, 0.);
    }

    template<typename T>
    void verifyOrthogonal( const std::vector< T > &v1, const std::vector< T > &v2,
                           std::string message ){
        /*!
         * Raise ``std::runtime_error`` exception if two vectors are not orthogonal
         *
         * \param &v1: The first vector
         * \param &v2: The second vector
         * \param message: Message for the ``std::runtime_error``
         */
        TARDIGRADE_ERROR_TOOLS_CATCH(
            if ( !isOrthogonal( v1, v2 ) ){
                throw std::runtime_error( message );
            }
        )

    }

    template<typename T>
    void verifyLength( const std::vector< T > &verifyVector, const unsigned int &expectedLength,
                       std::string message ){
        /*
         * Raise a ``std::length_error`` exception if the provided vector doesn't match the expected length.
         *
         * \param &verifyVector: The vector to check for length
         * \param &expectedLength: The expected vector length
         * \param &message: An optional message for the ``std::length_error`` exception
         */
        TARDIGRADE_ERROR_TOOLS_CATCH(
            if ( verifyVector.size( ) != expectedLength ){
                throw std::length_error( message );
            }
        )
    }

    template<typename T>
    void verifyLength( const std::vector< T > &verifyVectorOne,
                       const std::vector< T > &verifyVectorTwo,
                       std::string message ){
        /*
         * Raise a ``std::length_error`` exception if the provided vectors don't have matching lengths.
         *
         * \param &verifyVectorOne: The vector to check for length
         * \param &verifyVectorTwo: The vector to compare against
         * \param &message: An optional message for the ``std::length_error`` exception
         */
        TARDIGRADE_ERROR_TOOLS_CATCH(
            if ( verifyVectorOne.size( ) != verifyVectorTwo.size( ) ){
                throw std::length_error( message );
            }
        )
    }

    template<typename T>
    void verifyLength( const std::vector< std::vector< T > > &verifyVectorOne,
                       const std::vector< std::vector< T > > &verifyVectorTwo,
                       std::string message ){
        /*
         * Raise a ``std::length_error`` exception if the provided vectors don't have matching sizes.
         *
         * \param &verifyVectorOne: The vector to check for length
         * \param &verifyVectorTwo: The vector to compare against
         * \param &message: An optional message for the ``std::length_error`` exception
         */
        TARDIGRADE_ERROR_TOOLS_CATCH(
            if ( verifyVectorOne.size( ) != verifyVectorTwo.size( ) ){
                throw std::length_error( message );
            }
        )
        TARDIGRADE_ERROR_TOOLS_CATCH(
            for ( unsigned int row=0; row<verifyVectorOne.size( ); row++ ){
                verifyLength( verifyVectorOne[ row ], verifyVectorTwo[ row ] );
            }
        )
    }

    //Access Utilities
    template <typename T>
    int getValuesByIndex(const std::vector< T > &v, const std::vector< size_type > &indices,
        std::vector< T > &subv){
        /*!
         * Get the values of a vector referenced by index
         *
         * \param &v: The vector from which to retrieve get a sub-vector
         * \param &indices: The indices to retrieve
         * \param &subv: The subvector of values
         */

        //Resize subv
        subv.resize(indices.size());

        unsigned int i=0;
        for (auto it=indices.begin(); it!=indices.end(); it++, i++){
            subv[i] = v[*it];
        }
        return 0;
    }

    template <typename T>
    std::vector< T > getRow(const std::vector< T > &A, const unsigned int rows, const unsigned int cols, const unsigned int row){
        /*!
         * Get the row from the row-major storage matrix A
         *
         * \param &A: The row-major storage of matrix A
         * \param &rows: The number of rows in A
         * \param &cols: The number of columns in A
         * \param &row: The row to extract indexed from 0
         */

        TARDIGRADE_ERROR_TOOLS_CHECK( rows * cols == A.size( ), "Row-major matrix A's size is not consistent with the number of rows and columns" );

        TARDIGRADE_ERROR_TOOLS_CHECK( row < rows, "row cannot be greater than or equal to the number of rows" )

        std::vector< T > v( A.begin( ) + cols * row, A.begin( ) + cols * ( row + 1 ) );

        return v;
    }

    template <typename T>
    std::vector< T > getCol(const std::vector< T > &A, const unsigned int rows, const unsigned int cols, const unsigned int col){
        /*!
         * Get the column from the row-major storage matrix A
         *
         * \param &A: The row-major storage of matrix A
         * \param &rows: The number of rows in A
         * \param &cols: The number of columns in A
         * \param &col: The column to extract indexed from 0
         */

        TARDIGRADE_ERROR_TOOLS_CHECK( rows * cols == A.size( ), "Row-major matrix A's size is not consistent with the number of rows and columns" )

        TARDIGRADE_ERROR_TOOLS_CHECK( col < cols, "column cannot be greater than or equal to the number of columns" )

        std::vector< T > v( rows, 0 );

        for ( unsigned int i = 0; i < rows; i++ ){

            v[ i ] += A[ cols * i + col ];

        }

        return v;
    }

    //Appending Utilities
    template<typename T>
    std::vector< T > appendVectors(const std::vector< std::vector< T > > &A){
        /*!
         * Append a matrix into a row-major vector.
         *
         * \param &A: The matrix to be appended
         */

        std::vector< T > Avec;

        for (auto Ai=A.begin(); Ai!=A.end(); Ai++){
            Avec.insert(Avec.end(), (*Ai).begin(), (*Ai).end());
        }
        return Avec;
    }

    template<typename T>
    std::vector< T > appendVectors(const std::initializer_list< std::vector< T > > &list){
        /*!
         * Append a brace-enclosed initializer list to a row-major vector
         *
         * \param list: The list of vectors to append
         */

        std::vector< T > Avec;
        for (auto li=list.begin(); li!=list.end(); li++){
            Avec.insert(Avec.end(), (*li).begin(), (*li).end());
        }
        return Avec;
    }

    template< typename T >
    std::vector< std::vector< T > > inflate( const std::vector< T > &Avec, const unsigned int &nrows, const unsigned int &ncols ){
        /*!
         * Inflate the provided row-major vector into a 2D matrix.
         *
         * \param &Avec: The matrix in row-major form.
         * \param &nrows: The number of rows in the matrix.
         * \param &ncols: The number of columns in the matrix.
         */

        TARDIGRADE_ERROR_TOOLS_CHECK( Avec.size() == nrows * ncols, "Avec is not a consistent size with the desired dimensions of the matrix")

        std::vector< std::vector< T > > A( nrows, std::vector< T >( ncols ) );

        for ( unsigned int i = 0; i < nrows; i++ ){
            for ( unsigned int j = 0; j < ncols; j++ ){
                A[i][j] = Avec[ i * ncols + j ];
            }
        }
        return A;
    }

    //Sorting Utilities
    template<typename T>
    std::vector< size_type > argsort(const std::vector< T > &v){
        /*!
         * Find the indices required to sort a vector
         * Code from: https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
         *
         * \param &v: The vector to get the index sorted properties.
         */

        // initialize original index locations
        std::vector< size_type > idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // sort indices based on comparing values in v
        std::sort(idx.begin(), idx.end(),
                  [&v](size_type i1, size_type i2) {return v[i1] < v[i2];});

        return idx;
    }

    //Printing Utilities
    template<typename T>
    int print(const std::vector< T > &v){
        /*!
         * Print the contents of the vector to the terminal assuming << has been defined for each component
         *
         * \param &v: The vector to be displayed
         */

        for (auto it = v.begin(); it!=v.end(); it++){
            std::cout << *it << " ";
        }
        std::cout << "\n";
        return 0;
    }

    template<typename T>
    int print(const std::vector< std::vector< T > > &A){
        /*!
         * Print the contents of the matrix to the terminal assuming << has been defined for each component
         *
         * \param &A: The matrix to be displayed
         */

        for (auto it = A.begin(); it!=A.end(); it++){
            print(*it);
        }
        return 0;
    }

    template< typename T >
    int rotationMatrix( const std::vector< T > &bungeEulerAngles, std::vector< T > &directionCosines ){
        /*!
         * Calculate the pre-multiplying direction cosines rotation matrix from Euler angles - Bunge convention:
         *
         * 1. rotate around z-axis: \f$ \alpha \f$
         * 2. rotate around new x'-axis: \f$ \beta \f$
         * 3. rotate around new z'-axis: \f$ \gamma \f$
         *
         * Conventions:
         *
         * * Premultiply column vectors, \f$ v' = Rv \f$. Implies post-muliplying for row vectors, \f$ v' = vR \f$
         * * Represent active rotation. Returns rotated vectors defined in the original reference frame coordinate
         *   system.
         *
         * Used as:
         *
         * * Rotate a vector *defined in a fixed coordinate system* to a new, rotated vector *in the same fixed
         *   coordinate system* as \f$v'_{i} = R_{ij} v_{j}f\$ or \f$v' = Rv\f$
         * * Define a *fixed vector* in a new coordinate system by rotating the old coordinate system as
         *   \f$v'_{j} = R_{ji} v_{j}\f$ or \f$v' = R^{T}v\f$
         *
         * \param &bungeEulerAngles: Vector containing three Bunge-Euler angles in radians
         * \param &directionCosines: Row-major vector containing the 3x3 rotation matrix
         */

        std::vector< std::vector< T > > matrix;
        int return_value;
        return_value = rotationMatrix( bungeEulerAngles, matrix );
        directionCosines = appendVectors( matrix );
        return return_value;
    }

    template< typename T >
    int rotationMatrix( const std::vector< T > &bungeEulerAngles, std::vector< std::vector< T > > &directionCosines ){
        /*!
         * Calculate the pre-multiplying direction cosines rotation matrix from Euler angles - Bunge convention:
         *
         * 1. rotate around z-axis: \f$ \alpha \f$
         * 2. rotate around new x'-axis: \f$ \beta \f$
         * 3. rotate around new z'-axis: \f$ \gamma \f$
         *
         * Conventions:
         *
         * * Premultiply column vectors, \f$ v' = Rv \f$. Implies post-muliplying for row vectors, \f$ v' = vR \f$
         * * Represent active rotation. Returns rotated vectors defined in the original reference frame coordinate
         *   system.
         *
         * Used as:
         *
         * * Rotate a vector *defined in a fixed coordinate system* to a new, rotated vector *in the same fixed
         *   coordinate system* as \f$v'_{i} = R_{ij} v_{j}f\$ or \f$v' = Rv\f$
         * * Define a *fixed vector* in a new coordinate system by rotating the old coordinate system as
         *   \f$v'_{j} = R_{ji} v_{j}\f$ or \f$v' = R^{T}v\f$
         *
         * \param &bungeEulerAngles: Vector containing three Bunge-Euler angles in radians
         * \param &directionCosines: Matrix containing the 3x3 rotation matrix
         */
        TARDIGRADE_ERROR_TOOLS_CHECK( bungeEulerAngles.size( ) == ( 3 ), "There must be exactly three (3) Bunge-Euler angles." )

        double s1 = std::sin( bungeEulerAngles[ 0 ] );
        double c1 = std::cos( bungeEulerAngles[ 0 ] );
        double s2 = std::sin( bungeEulerAngles[ 1 ] );
        double c2 = std::cos( bungeEulerAngles[ 1 ] );
        double s3 = std::sin( bungeEulerAngles[ 2 ] );
        double c3 = std::cos( bungeEulerAngles[ 2 ] );

        directionCosines = { { c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1,  s1*s2 },
                             { c3*s1+c1*c2*s3, -s1*s3+c1*c2*c3, -c1*s2 },
                             {          s2*s3,           c3*s2,     c2 } };
        return 0;
    }

    template< typename T >
    int rotationMatrix( const std::vector< T > &bungeEulerAngles, std::vector< std::vector< T > > &directionCosines,
                        std::vector< std::vector< T > > &dDirectionCosinesdAlpha,
                        std::vector< std::vector< T > > &dDirectionCosinesdBeta,
                        std::vector< std::vector< T > > &dDirectionCosinesdGamma ){
        /*!
         * Calculate the pre-multiplying direction cosines rotation matrix from Euler angles - Bunge convention:
         *
         * 1. rotate around z-axis: \f$ \alpha \f$
         * 2. rotate around new x'-axis: \f$ \beta \f$
         * 3. rotate around new z'-axis: \f$ \gamma \f$
         *
         * Conventions:
         *
         * * Premultiply column vectors, \f$ v' = Rv \f$. Implies post-muliplying for row vectors, \f$ v' = vR \f$
         * * Represent active rotation. Returns rotated vectors defined in the original reference frame coordinate
         *   system.
         *
         * Used as:
         *
         * * Rotate a vector *defined in a fixed coordinate system* to a new, rotated vector *in the same fixed
         *   coordinate system* as \f$v'_{i} = R_{ij} v_{j}f\$ or \f$v' = Rv\f$
         * * Define a *fixed vector* in a new coordinate system by rotating the old coordinate system as
         *   \f$v'_{j} = R_{ji} v_{j}\f$ or \f$v' = R^{T}v\f$
         *
         * \param &bungeEulerAngles: Vector containing three Bunge-Euler angles in radians
         * \param &directionCosines: Matrix containing the 3x3 rotation matrix
         * \param &dDirectionCosinesdAlpha: Matrix partial derivative of the rotation matrix with respect to the first
         *     Euler angle: \f$ \alpha \f$.
         * \param &dDirectionCosinesdAlpha: Matrix partial derivative of the rotation matrix with respect to the second
         *     Euler angle: \f$ \beta \f$.
         * \param &dDirectionCosinesdGamma: Matrix partial derivative of the rotation matrix with respect to the third
         *     Euler angle: \f$ \gamma \f$.
         */

        double s1 = std::sin( bungeEulerAngles[ 0 ] );
        double c1 = std::cos( bungeEulerAngles[ 0 ] );
        double s2 = std::sin( bungeEulerAngles[ 1 ] );
        double c2 = std::cos( bungeEulerAngles[ 1 ] );
        double s3 = std::sin( bungeEulerAngles[ 2 ] );
        double c3 = std::cos( bungeEulerAngles[ 2 ] );

        int return_value;
        return_value = rotationMatrix( bungeEulerAngles, directionCosines );

        dDirectionCosinesdAlpha = { { -s1*c3-c1*c2*s3,  s1*s3-c1*c2*c3, c1*s2 },
                                    {  c1*c3-s1*c2*s3, -s1*c2*c3-c1*s3, s1*s2 },
                                    {              0.,              0.,    0. } };

        dDirectionCosinesdBeta = { {  s2*s1*s3,  s2*c3*s1,  c2*s1 },
                                   { -s2*c1*s3, -s2*c1*c3, -c1*c2 },
                                   {     c2*s3,     c2*c3,    -s2 } };

        dDirectionCosinesdGamma = { { -c1*s3-c2*s1*c3, -c1*c3+c2*s3*s1, 0. },
                                    { -s3*s1+c1*c2*c3, -c1*c2*s3-s1*c3, 0. },
                                    {           s2*c3,          -s3*s2, 0. } };

        return return_value;
    }

    template<typename T>
    void computeMatrixExponential( const std::vector< T > &A, const unsigned int &dim, std::vector< T > &expA, const unsigned int nmax, double tola, double tolr ){
        /*!
         * Compute the matrix exponential
         * 
         * \param &A: The matrix to compute the exponential of
         * \param &dim: The number of rows and columns in A
         * \param &expA: The matrix exponential of A
         * \param nmax: The maximum number of allowable iterations
         * \param tola: The absolute tolerance
         * \param tolr: The relative tolerance
         */

        TARDIGRADE_ERROR_TOOLS_CHECK( A.size( ) == dim * dim, "The matrix A's size is inconsistent with the dimension\n  A.size( ): " + std::to_string( A.size( ) ) +
                                                              "\n  dim * dim: " + std::to_string( dim ) + "\n" )

        std::vector< T > X( dim * dim, 0 );
        for ( unsigned int i = 0; i < dim; i++ ){ X[ dim * i + i ] = 1; }

        expA = X;

        double tol = tola * std::fabs( l2norm( A ) ) + tolr;

        for ( unsigned int n = 1; n < nmax; n++ ){

            std::vector< T > Xn( dim * dim, 0 );

            for ( unsigned int i = 0; i < dim; i++ ){

                for ( unsigned int j = 0; j < dim; j++ ){

                    for ( unsigned int k = 0; k < dim; k++ ){

                        Xn[ dim * i + k ] += X[ dim * i + j ] * A[ dim * j + k ];

                    }

                }

            }

            expA += Xn / std::tgamma( n + 1 );

            double delta = l2norm( Xn ) / std::tgamma( n + 1 );

            if ( delta < tol ){

                break;

            }

            X = Xn;

        }

    }

    template<typename T>
    void computeMatrixExponential( const std::vector< T > &A, const unsigned int &dim, std::vector< T > &expA, std::vector< T > &dExpAdA, const unsigned int nmax, double tola, double tolr ){
        /*!
         * Compute the matrix exponential
         * 
         * \param &A: The matrix to compute the exponential of
         * \param &dim: The number of rows and columns in A
         * \param &expA: The matrix exponential of A
         * \param &dExpAdA: The derivative of the matrix exponential of A w.r.t. A
         * \param nmax: The maximum number of allowable iterations
         * \param tola: The absolute tolerance
         * \param tolr: The relative tolerance
         */

        TARDIGRADE_ERROR_TOOLS_CHECK( A.size( ) == dim * dim, "The matrix A's size is inconsistent with the dimension\n  A.size( ): " + std::to_string( A.size( ) ) +
                                                              "\n  dim * dim: " + std::to_string( dim ) + "\n" )

        std::vector< T > X( dim * dim, 0 );
        for ( unsigned int i = 0; i < dim; i++ ){ X[ dim * i + i ] = 1; }

        expA = X;

        dExpAdA = std::vector< T >( dim * dim * dim * dim, 0 );

        std::vector< T > dXdA( dim * dim * dim * dim, 0 );

        double tol = tola * std::fabs( l2norm( A ) ) + tolr;

        for ( unsigned int n = 1; n < nmax; n++ ){

            std::vector< T > Xn( dim * dim, 0 );

            std::vector< T > dXndA( dim * dim * dim * dim, 0 );

            for ( unsigned int i = 0; i < dim; i++ ){

                for ( unsigned int j = 0; j < dim; j++ ){

                    for ( unsigned int k = 0; k < dim; k++ ){

                        Xn[ dim * i + k ] += X[ dim * i + j ] * A[ dim * j + k ];

                        dXndA[ dim * dim * dim * i + dim * dim * j + dim * k + j ] += X[ dim * i + k ];

                        for ( unsigned int ab = 0; ab < dim * dim; ab++ ){

                            dXndA[ dim * dim * dim * i + dim * dim * k + ab ] += dXdA[ dim * dim * dim * i + dim * dim * j + ab ] * A[ dim * j + k ];

                        }

                    }

                }

            }

            expA += Xn / std::tgamma( n + 1 );

            dExpAdA += dXndA / std::tgamma( n + 1 );

            double delta = l2norm( Xn ) / std::tgamma( n + 1 );

            if ( delta < tol ){

                break;

            }

            X = Xn;
            dXdA = dXndA;

        }

    }
    
    template<typename T>
    void computeMatrixExponentialScalingAndSquaring( const std::vector< T > &A, const unsigned int &dim, std::vector< T > &expA, const unsigned int nmax, double tola, double tolr ){
        /*!
         * Compute the matrix exponential using scaling-and-squaring
         * 
         * \param &A: The matrix to compute the exponential of
         * \param &dim: The number of rows and columns in A
         * \param &expA: The matrix exponential of A
         * \param nmax: The maximum number of allowable iterations
         * \param tola: The absolute tolerance
         * \param tolr: The relative tolerance
         */

        double normA = l2norm( A );

        unsigned int m = std::max( ( unsigned int )( std::ceil( std::sqrt( normA ) ) + 0.5 ), ( unsigned int )( 1 ) );

        std::vector< T > expAoverm;

        TARDIGRADE_ERROR_TOOLS_CATCH( computeMatrixExponential( A / m, dim, expAoverm, nmax, tola, tolr ) )

        expA = std::vector< T >( dim * dim, 0 );
        for ( unsigned int i = 0; i < dim; i++ ){ expA[ dim * i + i ] = 1; }

        for ( unsigned int i = 0; i < m; i++ ){

            std::vector< T > expAi( dim * dim, 0 );

            for ( unsigned int j = 0; j < dim; j++ ){

                for ( unsigned int k = 0; k < dim; k++ ){

                    for ( unsigned int l = 0; l < dim; l++ ){

                        expAi[ dim * j + l ] += expA[ dim * j + k ] * expAoverm[ dim * k + l ];

                    }

                }

            }

            expA = expAi;

        }

    }

    template<typename T>
    void computeMatrixExponentialScalingAndSquaring( const std::vector< T > &A, const unsigned int &dim, std::vector< T > &expA, std::vector< T > & dExpAdA, const unsigned int nmax, double tola, double tolr ){
        /*!
         * Compute the matrix exponential using scaling-and-squaring
         * 
         * \param &A: The matrix to compute the exponential of
         * \param &dim: The number of rows and columns in A
         * \param &expA: The matrix exponential of A
         * \param &dExpAdA: The derivative of the matrix exponential of A w.r.t. A
         * \param nmax: The maximum number of allowable iterations
         * \param tola: The absolute tolerance
         * \param tolr: The relative tolerance
         */

        double normA = l2norm( A );

        unsigned int m = std::max( ( unsigned int )( std::ceil( std::sqrt( normA ) ) + 0.5 ), ( unsigned int )( 1 ) );

        std::vector< T > expAoverm;

        std::vector< T > dExpAovermdA;

        TARDIGRADE_ERROR_TOOLS_CATCH( computeMatrixExponential( A / m, dim, expAoverm, dExpAovermdA, nmax, tola, tolr ) )

        dExpAovermdA /= m;

        expA = std::vector< T >( dim * dim, 0 );
        for ( unsigned int i = 0; i < dim; i++ ){ expA[ dim * i + i ] = 1; }

        dExpAdA = std::vector< T >( dim * dim * dim * dim, 0 );

        for ( unsigned int i = 0; i < m; i++ ){

            std::vector< T > expAi( dim * dim, 0 );

            std::vector< T > dExpAidA( dim * dim * dim * dim, 0 );

            for ( unsigned int j = 0; j < dim; j++ ){

                for ( unsigned int k = 0; k < dim; k++ ){

                    for ( unsigned int l = 0; l < dim; l++ ){

                        expAi[ dim * j + l ] += expA[ dim * j + k ] * expAoverm[ dim * k + l ];

                        for ( unsigned int ab = 0; ab < dim * dim; ab++ ){

                            dExpAidA[ dim * dim * dim * j + dim * dim * l + ab ] += dExpAdA[ dim * dim * dim * j + dim * dim * k + ab ] * expAoverm[ dim * k + l ]
                                                                                  + expA[ dim * j + k ] * dExpAovermdA[ dim * dim * dim * k + dim * dim * l + ab ];

                        }

                    }

                }

            }

            expA = expAi;

            dExpAdA = dExpAidA;

        }

    }

    #ifdef USE_EIGEN
        template< typename T >
        std::vector< double > solveLinearSystem( const std::vector< std::vector< T > > &A, const std::vector< T > &b,
            unsigned int &rank ){
            /*!
             * Solve a linear system of equations using Eigen. Note this uses a dense solver.
             *
             * \f$Ax = b\f$
             *
             * \param &A: The \f$A\f$ matrix
             * \param &b: The \f$b\f$ vector
             * \param &rank: The rank of \f$A\f$
             */

            //Get the number of rows in A
            unsigned int nrows = A.size( );

            //Append all of the vectors into one long vector
            const std::vector< T > Avec = appendVectors( A );

            unsigned int ncols = Avec.size( ) / nrows;
            TARDIGRADE_ERROR_TOOLS_CHECK( ( Avec.size( ) % nrows ) == 0, "A is not a regular matrix" )

            return solveLinearSystem( Avec, b, nrows, ncols, rank );

        }

        template< typename T >
        std::vector< double > solveLinearSystem(const std::vector< std::vector< T > > &A, const std::vector< T > &b,
            unsigned int &rank, solverType< T > &linearSolver ){
            /*!
             * Solve a linear system of equations using Eigen. Note this uses a dense solver.
             *
             * \f$Ax = b\f$
             *
             * \param &A: The \f$A\f$ matrix
             * \param &b: The \f$b\f$ vector
             * \param &rank: The rank of \f$A\f$
             * \param &linearSolver: The linear solver which contains the decomposed
             *     A matrix ( after the solve ). This is useful for when further
             *     non-linear solves are required such as in the construction
             *     of Jacobians of non-linear equaquations which were solved
             *     using Newton methods.
             */

            //Get the number of rows in A
            unsigned int nrows = A.size( );

            //Append all of the vectors into one long vector
            const std::vector< T > Avec = appendVectors( A );

            unsigned int ncols = Avec.size( ) / nrows;
            TARDIGRADE_ERROR_TOOLS_CHECK( ( Avec.size( ) % nrows ) == 0, "A is not a regular matrix" )

            return solveLinearSystem( Avec, b, nrows, ncols, rank, linearSolver );

        }

        template< typename T >
        std::vector< double > solveLinearSystem( const std::vector< T > &Avec, const std::vector< T > &b,
            const unsigned int nrows, const unsigned int ncols, unsigned int &rank ){
            /*!
             * Solve a linear system of equations using Eigen. Note this uses a dense solver.
             *
             * \f$Ax = b\f$
             *
             * \param &Avec: The vector form of the \f$A\f$ matrix ( row major )
             * \param &b: The \f$b\f$ vector
             * \param nrows: The number of rows of \f$A\f$
             * \param ncols: The number of columns of \f$A\f$
             * \param &rank: The rank of \f$A\f$
             */

            solverType< T > linearSolver;
            return solveLinearSystem( Avec, b, nrows, ncols, rank, linearSolver );
        }

        template< typename T >
        std::vector< double > solveLinearSystem( const std::vector< T > &Avec, const std::vector< T > &b,
            const unsigned int nrows, const unsigned int ncols, unsigned int &rank,
            solverType< T > &linearSolver ){
            /*!
             * Solve a linear system of equations using Eigen. Note this uses a dense solver.
             *
             * \f$Ax = b\f$
             *
             * \param &Avec: The vector form of the \f$A\f$ matrix ( row major )
             * \param &b: The \f$b\f$ vector
             * \param nrows: The number of rows of \f$A\f$
             * \param ncols: The number of columns of \f$A\f$
             * \param &rank: The rank of \f$A\f$
             * \param &linearSolver: The linear solver which contains the decomposed
             *     A matrix ( after the solve ). This is useful for when further
             *     non-linear solves are required such as in the construction
             *     of Jacobians of non-linear equaquations which were solved
             *     using Newton methods.
             */

            TARDIGRADE_ERROR_TOOLS_CHECK( Avec.size( ) == ( nrows * ncols ), "The size of Avec and the dimensions nrows and ncols do not align." )

            TARDIGRADE_ERROR_TOOLS_CHECK( b.size( ) == ncols, "The b vector's size is not consistent with A's dimension" )

            //Set up the Eigen maps for A and b
            Eigen::Map< const Eigen::Matrix< T, -1, -1, Eigen::RowMajor > > Amat( Avec.data( ), nrows, ncols );
            Eigen::Map< const Eigen::Matrix< T, -1,  1 > > bmat( b.data( ), nrows, 1 );

            //Set up the Eigen maps for the solution vector
            std::vector< double > x( nrows );
            Eigen::Map< Eigen::MatrixXd > xmat( x.data( ), nrows, 1 );

            //Perform the decomposition
            linearSolver = solverType< T >( Amat );

            rank = linearSolver.rank( );

            xmat = linearSolver.solve( bmat );
            return x;
        }

        template<typename T>
        T determinant(const std::vector< T > &Avec, const unsigned int nrows, const unsigned int ncols){
            /*!
             * Compute the determinant of the matrix A
             *
             * \param &Avec: The vector form of the A matrix (row major)
             * \param nrows: The number of rows
             * \param ncols: The number of columns
             */

            if (Avec.size() != (nrows*ncols)){
                std::cerr << "Error: The size of Avec and the dimensions nrows and ncols do not align.\n";
                assert(1==0);
            }

            //Set up the Eigen map for A
            Eigen::Map < const Eigen::Matrix<T, -1, -1, Eigen::RowMajor> > Amat(Avec.data(), nrows, ncols);
            return Amat.determinant();
        }

        template<typename T>
        std::vector< double > inverse(const std::vector< T > &Avec, const unsigned int nrows, const unsigned int ncols){
            /*!
             * Compute the inverse of a matrix in row-major format
             *
             * \param &Avec: The vector form of the A matrix (row major)
             * \param nrows: The number of rows
             * \param ncols: The number of columns
             */

            TARDIGRADE_ERROR_TOOLS_CHECK( Avec.size() == (nrows*ncols), "The size of Avec and the dimensions nrows and ncols do not agree.\n  Avec.size( ): " + std::to_string( Avec.size( ) ) + "\n  nrows * ncols: " + std::to_string( nrows * ncols ) )

            TARDIGRADE_ERROR_TOOLS_CHECK( nrows == ncols, "Error: The number of rows must equal the number of columns.\n  nrows: " + std::to_string( nrows ) + "\n  ncols: " + std::to_string(ncols) )

            //Set up the Eigen map for A
            Eigen::Map < const Eigen::Matrix<T, -1, -1, Eigen::RowMajor> > Amat(Avec.data(), nrows, ncols);

            //Set up the Eigen map for the inverse
            std::vector< double > AinvVec(nrows*ncols);
            Eigen::Map< Eigen::MatrixXd > Ainv(AinvVec.data(), ncols, nrows);

            //Compute the inverse
            Ainv = Amat.inverse().transpose(); //Note transpose because of how Eigen works

            return AinvVec;
        }

        template<typename T>
        std::vector< std::vector< double > > inverse( const std::vector< std::vector< T > > &A ){
            /*!
             * Compute the inverse of a matrix
             *
             * \param &A: The vector form of the A matrix
             */

            unsigned int nrows = A.size();
            unsigned int ncols;

            TARDIGRADE_ERROR_TOOLS_CHECK( nrows != 0, "A has no size" )

            ncols = A[0].size( );

            TARDIGRADE_ERROR_TOOLS_CHECK( ncols != 0, "A has no columns")

            std::vector< T > Avec = appendVectors( A );
            std::vector< double > Ainvvec = inverse( Avec, nrows, ncols );

            return inflate( Ainvvec, nrows, ncols );
        }

        template<typename T>
        std::vector< std::vector< double > > computeDInvADA( const std::vector< T > &invA, const unsigned int nrows, const unsigned int ncols ){
            /*!
             * Compute the derivative of the inverse of a matrix w.r.t. the matrix
             * 
             * \param &invA: The vector form of the inverse of the A matrix
             * \param nrows: The number of rows
             * \param ncols: The number of columns
             */

            return tardigradeVectorTools::inflate( computeFlatDInvADA( invA, nrows, ncols ), nrows * ncols, nrows * ncols );

        }
        template<typename T>
        std::vector< double > computeFlatDInvADA( const std::vector< T > &invA, const unsigned int nrows, const unsigned int ncols ){
            /*!
             * Compute the derivative of the inverse of a matrix w.r.t. the matrix
             * 
             * \param &invA: The vector form of the inverse of the A matrix
             * \param nrows: The number of rows
             * \param ncols: The number of columns
             */

            TARDIGRADE_ERROR_TOOLS_CHECK( invA.size() == (nrows*ncols), "The size of Avec and the dimensions nrows and ncols do not agree.\n  Avec.size( ): " + std::to_string( invA.size( ) ) + "\n  nrows * ncols: " + std::to_string( nrows * ncols ) )

            TARDIGRADE_ERROR_TOOLS_CHECK( nrows == ncols, "Error: The number of rows must equal the number of columns.\n  nrows: " + std::to_string( nrows ) + "\n  ncols: " + std::to_string(ncols) )

            std::vector< double > result( nrows * ncols * nrows * ncols, 0 );

            for ( unsigned int i = 0; i < nrows; i++ ){

                for ( unsigned int j = 0; j < ncols; j++ ){

                    for ( unsigned int a = 0; a < nrows; a++ ){

                        for ( unsigned int b = 0; b < ncols; b++ ){

                            result[ ncols * nrows * ncols * i + nrows * ncols * j + nrows * a + b ] = -invA[ ncols * i + a ] * invA[ ncols * b + j ];

                        }

                    }

                }

            }

            return result;

        }

        template<typename T>
        std::vector< double > computeDDetADA(const std::vector< T > &Avec, const unsigned int nrows, const unsigned int ncols){
            /*!
             * Compute the derivative of the determinant of a matrix w.r.t. the matrix
             *
             * \param &Avec: The matrix in vector form.
             * \param nrows: The number of rows in A
             * \param ncols: The number of columns in A
             */

            //Set up the Eigen map for A
            Eigen::Map < const Eigen::Matrix<T, -1, -1, Eigen::RowMajor> > Amat(Avec.data(), nrows, ncols);

            T detA = Amat.determinant();

            //Set up the Eigen map for the inverse
            std::vector< double > ddetAdA(nrows*ncols);
            Eigen::Map< Eigen::MatrixXd > ddetAdAmat(ddetAdA.data(), ncols, nrows);

            //Compute the derivative
            ddetAdAmat = detA * Amat.inverse(); //Note lack of transpose because of how Eigen works

            return ddetAdA;

        }

        template< typename T >
        std::vector< T > matrixMultiply(const std::vector< T > &A, const std::vector< T > &B,
                                        const unsigned int Arows, const unsigned int Acols,
                                        const unsigned int Brows, const unsigned int Bcols,
                                        const bool Atranspose, const bool Btranspose){
            /*!
             * Perform a matrix multiplication between two matrices stored in row-major format
             * $C_{ij} = A_{ik} B_{kj}$ if Atranspose = Btranspose = false
             * $C_{ij} = A_{ki} B_{kj}$ if Atranspose = true, Btranspose = false
             * $C_{ij} = A_{ik} B_{jk}$ if Atranspose = false, Btranspose = true
             * $C_{ij} = A_{ki} B_{jk}$ if Atranspose = true, Btranspose = true
             *
             * NOTE: The shape of B will be determined from the shape of A.
             *
             * \param &A: The A matrix in row-major format
             * \param &B: The B matrix in row-major format
             * \param Arows: The number of rows in A
             * \param Acols: The number of columns in A
             * \param Brows: The number of rows in B
             * \param Bcols: The number of columns in B
             * \param Atranspose: Boolean to indicate whether A should be transposed.
             * \param Btranspose: Boolean to indicate whether B should be transposed.
             */

            //Error handling
            TARDIGRADE_ERROR_TOOLS_CHECK( A.size() == Arows*Acols, "A has an incompatible shape")

            TARDIGRADE_ERROR_TOOLS_CHECK( B.size() == Brows*Bcols, "B has an incompatible shape")

            //Map A and B to Eigen matrices
            Eigen::Map < const Eigen::Matrix< T, -1, -1, Eigen::RowMajor > > Amat(A.data(), Arows, Acols);
            Eigen::Map < const Eigen::Matrix< T, -1, -1, Eigen::RowMajor > > Bmat(B.data(), Brows, Bcols);
            std::vector< T > C;

            //Perform the multiplication
            if ( Atranspose && Btranspose){
                C = std::vector< T >( Acols * Brows, 0);
                Eigen::Map < Eigen::Matrix< T, -1, -1, Eigen::RowMajor > > Cmat(C.data(), Acols, Brows);

                Cmat = Amat.transpose();
                Cmat *= Bmat.transpose();
            }
            else if ( Atranspose && !Btranspose){
                C = std::vector< T >( Acols * Bcols, 0);
                Eigen::Map < Eigen::Matrix< T, -1, -1, Eigen::RowMajor > > Cmat(C.data(), Acols, Bcols);

                Cmat = Amat.transpose() * Bmat;
            }
            else if ( !Atranspose && Btranspose){
                C = std::vector< T >( Arows * Brows, 0);
                Eigen::Map < Eigen::Matrix< T, -1, -1, Eigen::RowMajor > > Cmat(C.data(), Arows, Brows);

                Cmat = Amat * Bmat.transpose();
            }
            else{
                C  = std::vector< T >( Arows * Bcols, 0);
                Eigen::Map < Eigen::Matrix< T, -1, -1, Eigen::RowMajor > > Cmat(C.data(), Arows, Bcols);

                Cmat = Amat * Bmat;
            }

            return C;
        }

        template< typename T >
        int __matrixSqrtResidual(const std::vector< T > &A, const unsigned int Arows,
                                 const std::vector< T > &X,
                                 std::vector< double > &R, std::vector< std::vector< double > > &J){
            /*!
             * Compute the residual equation for the square root of a matrix.
             * This function is not intended to be accessed by the user.
             *
             * \param &A: The matrix A in row major form.
             * \param Arows: The number of rows in A.
             * \param &X: The estimate of the square root of A
             * \param &R: The value of the residual.
             * \param &J: The value of the jacobian.
             */

            Eigen::Map < const Eigen::Matrix< T, -1, -1, Eigen::RowMajor > > Amat(A.data(), Arows, Arows);
            Eigen::Map < const Eigen::Matrix< T, -1, -1, Eigen::RowMajor > > Xmat(X.data(), Arows, Arows);

            R = std::vector< double >(Arows*Arows, 0);
            Eigen::Map < Eigen::Matrix< double, -1, -1, Eigen::RowMajor > > Rmat(R.data(), Arows, Arows);

            Eigen::Matrix< double, -1, -1 > temp;

            std::vector< double > eyeVec(Arows*Arows);
            eye(eyeVec);

            J = std::vector< std::vector< double > >(Arows*Arows, std::vector< double >(Arows*Arows, 0 ) );
            temp = Xmat;
            temp *= Xmat;
            Rmat = Amat - temp;

            for (unsigned int i=0; i<Arows; i++){
                for (unsigned int j=0; j<Arows; j++){
                    for (unsigned int k=0; k<Arows; k++){
                         for (unsigned int l=0; l<Arows; l++){
                             J[Arows*i + j][Arows*k + l] = -eyeVec[Arows*i + k]*X[Arows*l + j]
                                                           -X[Arows*i + k]*eyeVec[Arows*j + l];
                         }
                    }
                }
            }
            return 0;
        }

        template< typename T >
        std::vector< double > matrixSqrt(const std::vector< T > &A, const unsigned int Arows,
                                    const double tolr, const double tola, const unsigned int maxIter,
                                    const unsigned int maxLS){
            /*!
             * Solve for the square root of the square matrix A.
             *
             * \param &A: The matrix A in row major form.
             * \param Arows: The number of rows in A.
             * \param &dAdX: The gradient of A w.r.t. X
             * \param tolr: The relative tolerance.
             * \param tola: The absolute tolerance.
             * \param maxIter: The maximum number of iterations
             * \param maxLS: The maximum number of line search iterations.
             */

            std::vector< std::vector< double > > dAdX;
            return matrixSqrt(A, Arows, dAdX, tolr, tola, maxIter);
        }

        template< typename T >
        std::vector< double > matrixSqrt(const std::vector< T > &A, const unsigned int Arows,
                                    std::vector< std::vector< double > > &dAdX,
                                    const double tolr, const double tola, const unsigned int maxIter,
                                    const unsigned int maxLS){
            /*!
             * Solve for the square root of the square matrix A.
             *
             * \param &A: The matrix A in row major form.
             * \param Arows: The number of rows in A.
             * \param &dAdX: The gradient of A w.r.t. X
             * \param tolr: The relative tolerance.
             * \param tola: The absolute tolerance.
             * \param maxIter: The maximum number of iterations
             * \param maxLS: The maximum number of line search iterations.
             */

            TARDIGRADE_ERROR_TOOLS_CHECK( A.size() == Arows * Arows, "A has an incompatible shape")

            //Initialize the output
            std::vector< double > X(Arows*Arows);
            eye(X);

            //Compute the first residual and jacobian
            std::vector< double > R;
            std::vector< std::vector< double > > J;

            __matrixSqrtResidual(A, Arows, X, R, J);

            double Rp = sqrt(dot(R, R));
            double Rnorm = Rp;
            double tol = tolr * Rp + tola;

            //Begin the Newton-Raphson loop
            unsigned int niter = 0;
            unsigned int rank;
            unsigned int nlsiter = 0;
            double lambda = 1.;
            std::vector< double > dX(Arows*Arows, 0);
            while ((Rp > tol) && (niter < maxIter)){
                dX = -solveLinearSystem(J, R, rank);
                TARDIGRADE_ERROR_TOOLS_CATCH(
                    if (rank < dX.size()){
                        std::cout << "niter: " << niter << "\n";
                        tardigradeVectorTools::print(J);
                        throw std::invalid_argument("J is rank defficent");
                    }
                )

                X += dX;

                __matrixSqrtResidual(A, Arows, X, R, J);

                Rnorm = sqrt(dot(R, R));

                lambda = 1;
                nlsiter = 0;
                while ((Rnorm > (1 - 1e-4)*Rp) && (nlsiter < maxLS)){

                    X -= lambda * dX;
                    lambda *= 0.5;
                    X += lambda * dX;

                    __matrixSqrtResidual(A, Arows, X, R, J);
                    Rnorm = sqrt(dot(R, R));

                    nlsiter++;

                }

                TARDIGRADE_ERROR_TOOLS_CATCH(
                    if (Rnorm > (1 - 1e-4)*Rp){
                        throw std::invalid_argument("Failure in line search");
                    }
                )

                Rp = Rnorm;

                niter++;

            }

            TARDIGRADE_ERROR_TOOLS_CATCH(
                if (Rp > tol){
                    throw std::invalid_argument("Matrix square root did not converge");
                }
            )

            //Set the jacobian
            dAdX = -J;

            return X;
        }

        template< typename T >
        void svd( const std::vector< T > &A, const unsigned int nrows, const unsigned int ncols,
                  std::vector< double > &U, std::vector< double > &Sigma,
                  std::vector< double > &V ){
            /*!
             * Compute the singular value decomposition of a real valued matrix A where A is of the form
             * A = U Sigma VT
             * where VT indicates the transpose.
             *
             * \param &A: The matrix in row-major format
             * \param nrows: The number of rows in A
             * \param ncols: The number of columns in A
             * \param &U: The returned left-hand side unitary matrix in row-major format
             * \param &Sigma: The singular values
             * \param &V: The returned right-hand side unitary matrix in row-major format
             */

            TARDIGRADE_ERROR_TOOLS_CHECK( A.size( ) == nrows * ncols, "A's size is not consistent with the indicated number of rows and columns" )

            // Clear and Re-size the output vectors
            U.clear( );
            U.resize( nrows * nrows );

            Sigma.clear( );
            Sigma.resize( std::min( nrows, ncols ) );

            V.clear( );
            V.resize( ncols * ncols );

            // Construct the Eigen Maps
            Eigen::Map< const Eigen::Matrix< T, -1, -1, Eigen::RowMajor > > _A( A.data(), nrows, ncols );

            Eigen::Map< Eigen::Matrix< T, -1, -1, Eigen::RowMajor > > _U( U.data(), nrows, nrows );

            Eigen::Map< Eigen::Matrix< T, -1,  -1, Eigen::RowMajor > > _Sigma( Sigma.data(), Sigma.size( ), 1 );

            Eigen::Map< Eigen::Matrix< T, -1, -1, Eigen::RowMajor > > _V( V.data(), ncols, ncols );

            // Perform the singular value decomposition
            Eigen::JacobiSVD< Eigen::Matrix< T, -1, -1, Eigen::RowMajor > > _svd( _A, Eigen::ComputeFullU | Eigen::ComputeFullV );

            _U = _svd.matrixU( );

            _Sigma = _svd.singularValues( );

            _V = _svd.matrixV( );

            return;

        }

        template< typename T >
        void polar_decomposition( const std::vector< T > &A, const unsigned int nrows, const unsigned int ncols,
                                  std::vector< double > &R, std::vector< double > &U, const bool left ){
            /*!
             * Perform the polar decomposition of the matrix \f$A\f$. If left is false the decomposition will be:
             *
             * \f$A = R U\f$
             *
             * If left is true the decomposition will be:
             *
             * \f$A = U R\f$
             *
             * /param &A: The matrix to be decomposed
             * /param &nrows: The number of rows in A
             * /param &ncols: The number of columns in A
             * /param &R: The rotation tensor
             * /param &U: The stretch tensor. Left or right stretch is determined by the parameter `left`
             * /param &left: The flag indicating of the right decomposition (\f$A = RU\f$) or the left decomposition
             *     (\f$A = UR\f$) is to be performed.
             */

            // Compute Usqrd
            std::vector< double > Usqrd;
            unsigned int Urows;

            if ( left ){

                Usqrd = tardigradeVectorTools::matrixMultiply( A, A, nrows, ncols, ncols, nrows, 0, 1 );
                Urows = nrows;

            }
            else{

                Usqrd = tardigradeVectorTools::matrixMultiply( A, A, ncols, nrows, nrows, ncols, 1, 0 );
                Urows = ncols;

            }

            // Perform the matrix square root of Usqrd
            U = matrixSqrt( Usqrd, Urows );

            // Compute the rotation matrix
            std::vector< double > Uinv = tardigradeVectorTools::inverse( U, nrows, ncols );

            if ( left ){

                R = tardigradeVectorTools::matrixMultiply( Uinv, A, nrows, nrows, nrows, ncols, 0, 0 );

            }
            else{

                R = tardigradeVectorTools::matrixMultiply( A, Uinv, nrows, ncols, ncols, ncols, 0, 0 );

            }

            return;
        }

    #endif
}
