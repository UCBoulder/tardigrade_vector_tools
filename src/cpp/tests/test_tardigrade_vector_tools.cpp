/**
 * \file test_tardigrade_vector_tools.cpp
 *
 * Tests for tardigrade_vector_tools
 */

#include <math.h>

#include <fstream>
#include <iostream>
#include <vector>
#define USE_EIGEN
#include <tardigrade_vector_tools.h>

#define BOOST_TEST_MODULE test_tardigrade_vector_tools
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#define DEFAULT_TEST_TOLERANCE 1e-6
#define CHECK_PER_ELEMENT boost::test_tools::per_element()

typedef double                  floatType;
typedef std::vector<floatType>  vectorType;
typedef std::vector<vectorType> matrixType;

void print(vectorType a) {
    /*!
     * Print the vector to the terminal
     */

    for (unsigned int i = 0; i < a.size(); i++) {
        std::cout << a[i] << " ";
    }
    std::cout << "\n";
}

void print(matrixType A) {
    /*!
     * Print the matrix to the terminal
     */

    for (unsigned int i = 0; i < A.size(); i++) {
        print(A[i]);
    }
}

BOOST_AUTO_TEST_CASE(test_addition_operators, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the addition operators
     */

    vectorType a = {1, 2, 3};
    vectorType b = {-2, 7, 2};
    vectorType c;

    matrixType A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    matrixType B = {
        {-1, -3, -3},
        {2,  5,  6 }
    };
    matrixType C;

    a += b;

    vectorType a_answer_1 = {-1, 9, 5};
    vectorType c_answer_1 = {-3, 16, 7};

    BOOST_TEST(a == a_answer_1);

    c = a + b;

    BOOST_TEST(c == c_answer_1);

    a += 1.;

    vectorType a_answer_2 = {0, 10, 6};
    vectorType c_answer_2 = {2, 12, 8};
    vectorType A_answer   = {0, -1, 0, 6, 10, 12};
    vectorType C_answer   = {-1, -4, -3, 8, 15, 18};
    vectorType A_answer2  = {1, 0, 1, 7, 11, 13};

    BOOST_TEST(a == a_answer_2, CHECK_PER_ELEMENT);

    c = a + 2.;
    BOOST_TEST(c == c_answer_2, CHECK_PER_ELEMENT);
    BOOST_TEST(a == a_answer_2, CHECK_PER_ELEMENT);

    c = 2. + a;
    BOOST_TEST(c == c_answer_2, CHECK_PER_ELEMENT);
    BOOST_TEST(a == a_answer_2, CHECK_PER_ELEMENT);

    A += B;
    BOOST_TEST(tardigradeVectorTools::appendVectors(A) == A_answer, CHECK_PER_ELEMENT);

    C = A + B;
    BOOST_TEST(tardigradeVectorTools::appendVectors(C) == C_answer, CHECK_PER_ELEMENT);

    A += 1.;

    BOOST_TEST(tardigradeVectorTools::appendVectors(A) == A_answer2, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_subtraction_operators, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the subtraction operators
     */

    vectorType a = {1, 2, 3};
    vectorType b = {-2, 7, 2};
    vectorType c;

    matrixType A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    matrixType B = {
        {-1, -3, -3},
        {2,  5,  6 }
    };
    matrixType C;

    vectorType a_answer_1 = {-1, -2, -3};
    vectorType a_answer_2 = {3, -5, 1};
    vectorType a_answer_3 = {2, -6, 0};

    vectorType c_answer_1 = {5, -12, -1};
    vectorType c_answer_2 = {0, -8, -2};
    vectorType c_answer_3 = {0, 8, 2};

    vectorType A_answer_1 = {-1, -2, -3, -4, -5, -6};
    vectorType A_answer_2 = {2, 5, 6, 2, 0, 0};

    vectorType C_answer_1 = {3, 8, 9, 0, -5, -6};

    BOOST_TEST((-a) == a_answer_1, CHECK_PER_ELEMENT);

    a -= b;

    BOOST_TEST(a == a_answer_2, CHECK_PER_ELEMENT);

    c = a - b;

    BOOST_TEST(c == c_answer_1, CHECK_PER_ELEMENT);

    a -= 1.;

    BOOST_TEST(a == a_answer_3, CHECK_PER_ELEMENT);

    c = a - 2.;
    BOOST_TEST(c == c_answer_2, CHECK_PER_ELEMENT);
    BOOST_TEST(a == a_answer_3, CHECK_PER_ELEMENT);

    c = 2. - a;
    BOOST_TEST(c == c_answer_3, CHECK_PER_ELEMENT);
    BOOST_TEST(a == a_answer_3, CHECK_PER_ELEMENT);

    BOOST_TEST(tardigradeVectorTools::appendVectors(-A) == A_answer_1, CHECK_PER_ELEMENT);

    A -= B;
    BOOST_TEST(tardigradeVectorTools::appendVectors(A) == A_answer_2, CHECK_PER_ELEMENT);

    C = A - B;
    BOOST_TEST(tardigradeVectorTools::appendVectors(C) == C_answer_1, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_multiplication_operators, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the multiplication operators
     */

    vectorType a = {1, 2, 3};
    vectorType b, c;

    matrixType A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    matrixType B, C;

    vectorType a_answer = {2, 4, 6};
    vectorType b_answer = {6, 12, 18};

    matrixType A_answer = {
        {2, 4,  6 },
        {8, 10, 12}
    };
    matrixType B_answer = {
        {6,  12, 18},
        {24, 30, 36}
    };

    a *= 2;

    BOOST_TEST(a == a_answer, CHECK_PER_ELEMENT);

    b = 3 * a;
    c = a * 3;

    BOOST_TEST(b == c, CHECK_PER_ELEMENT);
    BOOST_TEST(b == b_answer, CHECK_PER_ELEMENT);

    A *= 2;

    BOOST_TEST(tardigradeVectorTools::appendVectors(A) == tardigradeVectorTools::appendVectors(A_answer),
               CHECK_PER_ELEMENT);

    B = 3 * A;
    C = A * 3;

    BOOST_TEST(tardigradeVectorTools::appendVectors(B) == tardigradeVectorTools::appendVectors(B_answer),
               CHECK_PER_ELEMENT);
    BOOST_TEST(tardigradeVectorTools::appendVectors(C) == tardigradeVectorTools::appendVectors(B_answer),
               CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_division_operators, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the division operators
     */

    vectorType a = {1, 2, 3};
    vectorType b;

    matrixType A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    matrixType B, C;

    vectorType a_answer = {0.50, 1.0, 1.50};
    vectorType b_answer = {0.25, 0.5, 0.75};

    matrixType A_answer = {
        {0.50, 1.00, 1.50},
        {2.00, 2.50, 3.00}
    };
    matrixType B_answer = {
        {0.25, 0.50, 0.75},
        {1.00, 1.25, 1.50}
    };

    a /= 2;

    BOOST_TEST(a == a_answer, CHECK_PER_ELEMENT);

    b = a / 2;

    BOOST_TEST(b == b_answer, CHECK_PER_ELEMENT);

    A /= 2;

    BOOST_TEST(tardigradeVectorTools::appendVectors(A) == tardigradeVectorTools::appendVectors(A_answer),
               CHECK_PER_ELEMENT);

    B = A / 2;

    BOOST_TEST(tardigradeVectorTools::appendVectors(B) == tardigradeVectorTools::appendVectors(B_answer),
               CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_computeMean, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    // boost::test_tools::tolerance( 0.1 ) ){
    /*!
     * Test the computation of the mean of a vector of vectors
     */

    matrixType A = {
        {1,  2,  3.0, 4},
        {-4, 13, 0.4, 5},
        {2,  6,  1.0, 7}
    };

    vectorType answer = {-1. / 3., 7., 8.8 / 6., 5. + 1 / 3.};

    vectorType result;
    tardigradeVectorTools::computeMean(A, result);

    BOOST_TEST(result == answer, CHECK_PER_ELEMENT);

    result = tardigradeVectorTools::computeMean(A);
    BOOST_TEST(result == answer, CHECK_PER_ELEMENT);

    vectorType A_flat = tardigradeVectorTools::appendVectors(A);
    result            = vectorType(4, 0.);
    tardigradeVectorTools::computeRowMajorMean<floatType>(std::begin(A_flat), std::end(A_flat), std::begin(result),
                                                          std::end(result));
    BOOST_TEST(result == answer, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_cross, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the cross product of two vectors
     */

    vectorType a = {1, 2};
    vectorType b = {-1, 7};
    vectorType c;

    vectorType c_answer_1 = {0, 0, 9};
    vectorType c_answer_2 = {-27, 0, 9};

    tardigradeVectorTools::cross(a, b, c);

    BOOST_TEST(tardigradeVectorTools::fuzzyEquals(c, c_answer_1));

    a = {1, 2, 3};
    b = {-1, 7, -3};

    tardigradeVectorTools::cross(a, b, c);

    BOOST_TEST(c == c_answer_2, CHECK_PER_ELEMENT);

    c = tardigradeVectorTools::cross(a, b);

    BOOST_TEST(c == c_answer_2, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_dot, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the dot product of two vectors
     */

    vectorType a = {1, 2, 3};
    vectorType b = {-1, 7, 6};
    floatType  c;

    floatType  c_answer_1 = -1 + 14 + 18;
    vectorType d_answer_1 = {14, 32, 50};
    vectorType C_answer_1 = {84, 90, 96, 201, 216, 231, 318, 342, 366};

    tardigradeVectorTools::dot(a, b, c);

    BOOST_TEST(c == c_answer_1);

    c = tardigradeVectorTools::dot(a, b);
    BOOST_TEST(c == c_answer_1);

    matrixType A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    vectorType d;

    d = tardigradeVectorTools::dot(A, a);

    BOOST_TEST(d == d_answer_1, CHECK_PER_ELEMENT);

    vectorType flat_A = tardigradeVectorTools::appendVectors(A);

    d = vectorType(3, 0.);

    tardigradeVectorTools::rowMajorDot<floatType>(std::begin(flat_A), std::end(flat_A), std::begin(a), std::end(a),
                                                  std::begin(d), std::end(d));

    BOOST_TEST(d == d_answer_1, CHECK_PER_ELEMENT);

    matrixType B = {
        {10, 11, 12},
        {13, 14, 15},
        {16, 17, 18}
    };

    vectorType flat_B = tardigradeVectorTools::appendVectors(B);

    matrixType C = tardigradeVectorTools::dot(A, B);

    vectorType flat_C(9, 0);

    BOOST_TEST(tardigradeVectorTools::appendVectors(C) == C_answer_1, CHECK_PER_ELEMENT);

    tardigradeVectorTools::rowMajorDot<floatType>(std::begin(flat_A), std::end(flat_A), std::begin(flat_B),
                                                  std::end(flat_B), 3, 3, std::begin(flat_C), std::end(flat_C));

    BOOST_TEST(flat_C == C_answer_1, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_dotT, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the dot product of two matrices
     */

    matrixType A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    matrixType B = {
        {10, 11, 12},
        {13, 14, 15}
    };

    vectorType answer = {68, 86, 167, 212, 266, 338};

    BOOST_TEST(tardigradeVectorTools::appendVectors(tardigradeVectorTools::dotT(A, B)) == answer, CHECK_PER_ELEMENT);

    vectorType flat_A = tardigradeVectorTools::appendVectors(A);

    vectorType flat_B = tardigradeVectorTools::appendVectors(B);

    vectorType flat_C(6, 0);

    tardigradeVectorTools::rowMajorDotT<floatType>(std::begin(flat_A), std::end(flat_A), std::begin(flat_B),
                                                   std::end(flat_B), 3, 2, std::begin(flat_C), std::end(flat_C));

    BOOST_TEST(flat_C == answer, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_Tdot, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the dot product of two matrices
     */

    matrixType A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    matrixType B = {
        {10, 11},
        {12, 13},
        {14, 15}
    };

    vectorType b = {4, 5, 6};

    vectorType matrixAnswer = {156, 168, 192, 207, 228, 246};

    vectorType vectorAnswer = {66, 81, 96};

    BOOST_TEST(tardigradeVectorTools::appendVectors(tardigradeVectorTools::Tdot(A, B)) == matrixAnswer,
               CHECK_PER_ELEMENT);

    BOOST_TEST(tardigradeVectorTools::Tdot(A, b) == vectorAnswer, CHECK_PER_ELEMENT);

    vectorType flat_A = tardigradeVectorTools::appendVectors(A);

    vectorType flat_B = tardigradeVectorTools::appendVectors(B);

    vectorType result(3, 0);

    tardigradeVectorTools::rowMajorTdot<floatType>(std::begin(flat_A), std::end(flat_A), std::begin(b), std::end(b),
                                                   std::begin(result), std::end(result));

    BOOST_TEST(result == vectorAnswer, CHECK_PER_ELEMENT);

    vectorType flat_C(6, 0);

    tardigradeVectorTools::rowMajorTDot<floatType>(std::begin(flat_A), std::end(flat_A), std::begin(flat_B),
                                                   std::end(flat_B), 3, 2, std::begin(flat_C), std::end(flat_C));

    BOOST_TEST(flat_C == matrixAnswer, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_TdotT, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the dot product of two matrices
     */

    matrixType A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    matrixType B = {
        {10, 11, 12},
        {13, 14, 15}
    };

    vectorType answer = {138, 174, 171, 216, 204, 258};

    BOOST_TEST(tardigradeVectorTools::appendVectors(tardigradeVectorTools::TdotT(A, B)) == answer, CHECK_PER_ELEMENT);

    vectorType flat_A = tardigradeVectorTools::appendVectors(A);

    vectorType flat_B = tardigradeVectorTools::appendVectors(B);

    vectorType flat_C(6);

    tardigradeVectorTools::rowMajorTdotT<floatType>(std::begin(flat_A), std::end(flat_A), std::begin(flat_B),
                                                    std::end(flat_B), 3, 2, std::begin(flat_C), std::end(flat_C));

    BOOST_TEST(flat_C == answer, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_inner, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    // Initialize test variables
    vectorType Avec, Bvec;
    Avec = Bvec = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    matrixType A, B;
    A = B = {
        {1., 0., 0.},
        {0., 1., 0.},
        {0., 0., 1.}
    };
    floatType expected = 3.;
    floatType result;

    // Test inner product of row major matrices
    result = 0.;
    tardigradeVectorTools::inner(Avec, Bvec, result);
    BOOST_TEST(result == expected);

    result = tardigradeVectorTools::inner(Avec, Bvec);
    BOOST_TEST(result == expected);

    // Test inner product of matrices
    result = 0.;
    tardigradeVectorTools::inner(A, B, result);
    BOOST_TEST(result == expected);

    result = tardigradeVectorTools::inner(A, B);
    BOOST_TEST(result == expected);
}

BOOST_AUTO_TEST_CASE(test_trace, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the trace for a square matrix
     */

    vectorType a = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    floatType  c;

    tardigradeVectorTools::trace(a, c);

    BOOST_TEST(tardigradeVectorTools::fuzzyEquals<floatType>(c, 3.));

    c = 0;
    tardigradeVectorTools::rowMajorTrace<3, 3>(std::begin(a), std::end(a), c);

    BOOST_TEST(c == 3.);

    // TODO: Refactor with boost or pytest
    vectorType b = {1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.};

    BOOST_CHECK_THROW(tardigradeVectorTools::trace(b, c), std::runtime_error);

    c = 0;
    tardigradeVectorTools::rowMajorTrace<4, 3>(std::begin(b), std::end(b), c);

    BOOST_TEST(c == 2.);

    c = tardigradeVectorTools::trace(a);
    BOOST_TEST(c == 3.);

    matrixType A = {
        {1., 0., 0.},
        {0., 1., 0.},
        {0., 0., 1.}
    };

    c = 0.;
    tardigradeVectorTools::trace(A, c);
    BOOST_TEST(c == 3.);

    c = tardigradeVectorTools::trace(A);
    BOOST_TEST(c == 3.);
}

BOOST_AUTO_TEST_CASE(test_l2norm, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the l2norm of vectors and matrices
     */

    vectorType a = {1, 2, 3};
    matrixType A = {
        {1, 2, 3, 4},
        {5, 6, 7, 8}
    };

    double r = tardigradeVectorTools::l2norm(a);
    BOOST_TEST(r == 3.7416573867739413);

    r = tardigradeVectorTools::l2norm(A);
    BOOST_TEST(r == 14.2828568570857);
}

BOOST_AUTO_TEST_CASE(test_unitVector, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of unit vectors
     */
    std::vector<std::vector<int> >    vector_int;
    std::vector<std::vector<double> > expected;
    std::vector<double>               vector_double;
    std::vector<double>               answer;

    double unit_cube = 1. / std::sqrt(3.);

    vector_int = {
        {1,  0,  0 },
        {0,  1,  0 },
        {0,  0,  1 },
        {-1, 0,  0 },
        {0,  -1, 0 },
        {0,  0,  -1},
        {1,  1,  1 },
        {2,  2,  2 },
        {-1, -1, -1},
        {-2, -2, -2}
    };
    expected = {
        {1,          0,          0         },
        {0,          1,          0         },
        {0,          0,          1         },
        {-1,         0,          0         },
        {0,          -1,         0         },
        {0,          0,          -1        },
        {unit_cube,  unit_cube,  unit_cube },
        {unit_cube,  unit_cube,  unit_cube },
        {-unit_cube, -unit_cube, -unit_cube},
        {-unit_cube, -unit_cube, -unit_cube}
    };

    for (unsigned int i = 0; i < vector_int.size(); i++) {
        answer = tardigradeVectorTools::unitVector(vector_int[i]);
        BOOST_TEST(answer == expected[i], boost::test_tools::per_element());

        vector_double = std::vector<double>(vector_int[i].begin(), vector_int[i].end());
        answer        = tardigradeVectorTools::unitVector(vector_double);
        BOOST_TEST(answer == expected[i], boost::test_tools::per_element());

        std::vector<double> unit_in_place = {(double)vector_int[i][0], (double)vector_int[i][1],
                                             (double)vector_int[i][2]};
        tardigradeVectorTools::unitVector<double>(std::begin(unit_in_place), std::end(unit_in_place));
        BOOST_TEST(unit_in_place == expected[i], boost::test_tools::per_element());
    }
}

BOOST_AUTO_TEST_CASE(test_argsort, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the utility that returns the indices required to sort a vector
     */

    vectorType                a   = {1, -2, 7, 3, 9, 11};
    std::vector<unsigned int> idx = tardigradeVectorTools::argsort(a);

    std::vector<unsigned int> answer = {1, 0, 3, 2, 4, 5};

    BOOST_TEST(idx == answer);
}

BOOST_AUTO_TEST_CASE(test_fuzzyEquals, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the tolerant compare function
     */

    matrixType a = {
        {1,   -2,  3,        2.4, 1e-9,      -1e-7},
        {1.4, 8.5, 1 + 1e-9, 4,   -2 + 1e-3, 100  }
    };

    BOOST_CHECK(tardigradeVectorTools::fuzzyEquals(a, {
                                                          {1,   -2,  3, 2.4, 0,                0         },
                                                          {1.4, 8.5, 1, 4,   -2 + 1e-3 + 1e-9, 100 + 1e-4}
    }));
}

BOOST_AUTO_TEST_CASE(test_equals, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the exact equality function
     */

    unsigned int                            a = 1;
    std::vector<unsigned int>               v = {1, 2, 3, 4};
    std::vector<std::vector<unsigned int> > m = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    BOOST_CHECK(tardigradeVectorTools::equals<unsigned int>(a, 1));

    BOOST_CHECK(!tardigradeVectorTools::equals<unsigned int>(a, 2));

    BOOST_CHECK(tardigradeVectorTools::equals(v, {1, 2, 3, 4}));

    BOOST_CHECK(!tardigradeVectorTools::equals(v, {1, 2, 2, 4}));

    BOOST_CHECK(tardigradeVectorTools::equals(m, {
                                                     {1, 2, 3},
                                                     {4, 5, 6},
                                                     {7, 8, 9}
    }));

    BOOST_CHECK(!tardigradeVectorTools::equals(m, {
                                                      {1, 2, 3},
                                                      {4, 5, 4},
                                                      {7, 8, 9}
    }));
}

BOOST_AUTO_TEST_CASE(test_verifyLength, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    std::vector<double> testVectorDouble = {1.};
    BOOST_CHECK_NO_THROW(tardigradeVectorTools::verifyLength(testVectorDouble, 1));
    BOOST_CHECK_THROW(tardigradeVectorTools::verifyLength(testVectorDouble, 2), std::runtime_error);

    std::vector<int> testVectorInt = {1};
    BOOST_CHECK_NO_THROW(tardigradeVectorTools::verifyLength(testVectorInt, 1));
    BOOST_CHECK_THROW(tardigradeVectorTools::verifyLength(testVectorInt, 2), std::runtime_error);

    std::vector<std::vector<double> > testNestedVectorDouble             = {{1.}, {2.}};
    std::vector<std::vector<double> > testNestedVectorDoubleLonger       = {{1.}, {2.}, {3.}};
    std::vector<std::vector<double> > testNestedVectorDoubleLongerRagged = {
        {1.},
        {2., 3.}
    };
    BOOST_CHECK_NO_THROW(tardigradeVectorTools::verifyLength(testNestedVectorDouble, testNestedVectorDouble));
    BOOST_CHECK_NO_THROW(tardigradeVectorTools::verifyLength(testNestedVectorDoubleLongerRagged,
                                                             testNestedVectorDoubleLongerRagged));
    BOOST_CHECK_THROW(tardigradeVectorTools::verifyLength(testNestedVectorDouble, testNestedVectorDoubleLonger),
                      std::runtime_error);
    BOOST_CHECK_THROW(tardigradeVectorTools::verifyLength(testNestedVectorDouble, testNestedVectorDoubleLongerRagged),
                      std::runtime_error);

    std::vector<std::vector<int> > testNestedVectorInt             = {{1}, {2}};
    std::vector<std::vector<int> > testNestedVectorIntLonger       = {{1}, {2}, {3}};
    std::vector<std::vector<int> > testNestedVectorIntLongerRagged = {
        {1},
        {2, 3}
    };
    BOOST_CHECK_NO_THROW(tardigradeVectorTools::verifyLength(testNestedVectorInt, testNestedVectorInt));
    BOOST_CHECK_NO_THROW(tardigradeVectorTools::verifyLength(testNestedVectorIntLongerRagged,
                                                             testNestedVectorIntLongerRagged));
    BOOST_CHECK_THROW(tardigradeVectorTools::verifyLength(testNestedVectorInt, testNestedVectorIntLonger),
                      std::runtime_error);
    BOOST_CHECK_THROW(tardigradeVectorTools::verifyLength(testNestedVectorInt, testNestedVectorIntLongerRagged),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(test_getValuesByIndex, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the retrieval of values of a vector as
     * indicated by vector of indices.
     */

    vectorType v = {1, 2, 3, 4, 5, 6};
    matrixType m = {
        {1,  2,  3 },
        {4,  5,  6 },
        {7,  8,  9 },
        {10, 11, 12},
        {13, 14, 15}
    };

    vectorType rv_answer = {2, 4, 1};
    vectorType rm_answer = {4, 5, 6, 10, 11, 12, 1, 2, 3};

    std::vector<tardigradeVectorTools::size_type> indices = {1, 3, 0};

    vectorType rv;
    tardigradeVectorTools::getValuesByIndex(v, indices, rv);
    BOOST_TEST(rv == rv_answer, CHECK_PER_ELEMENT);

    matrixType rm;
    tardigradeVectorTools::getValuesByIndex(m, indices, rm);
    BOOST_TEST(tardigradeVectorTools::appendVectors(rm) == rm_answer, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_getRow, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the retrieval of values of a row of the matrix
     */

    vectorType m = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    unsigned int row = 2;

    vectorType answer = {7, 8, 9};

    BOOST_TEST(tardigradeVectorTools::getRow(m, 5, 3, row) == answer, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_getCol, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the retrieval of values of a column of the matrix
     */

    vectorType m = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    unsigned int col = 1;

    vectorType answer = {2, 5, 8, 11, 14};

    vectorType result = tardigradeVectorTools::getCol(m, 5, 3, col);

    BOOST_TEST(tardigradeVectorTools::getCol(m, 5, 3, col) == answer, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_appendVectors, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the utility to append a vector of vectors into a row-major vector.
     */

    matrixType m = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    vectorType v = tardigradeVectorTools::appendVectors(m);

    vectorType v_answer_1 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    vectorType v_answer_2 = {1, 7, 5, 4, 6, 2};

    BOOST_TEST(v == v_answer_1, CHECK_PER_ELEMENT);

    v.clear();
    vectorType v1 = {1, 7, 5};
    vectorType v2 = {4, 6, 2};
    v             = tardigradeVectorTools::appendVectors({v1, v2});

    BOOST_TEST(v == v_answer_2, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_rotationMatrix, *boost::unit_test::tolerance(1.0e-15)) {
    /*!
     * Test the Bunge-Euler rotation matrix construction: Z-X'-Z'.
     */

    std::vector<std::vector<double> >               bungeEulerAngles;
    std::vector<std::vector<double> >               directionCosines;
    std::vector<double>                             directionCosinesVector;
    std::vector<std::vector<double> >               dDirectionCosinesdAlpha;
    std::vector<std::vector<double> >               dDirectionCosinesdBeta;
    std::vector<std::vector<double> >               dDirectionCosinesdGamma;
    std::vector<std::vector<std::vector<double> > > expectedDirectionCosines;
    std::vector<std::vector<std::vector<double> > > expected_dDirectionCosinesdAlpha;
    std::vector<std::vector<std::vector<double> > > expected_dDirectionCosinesdBeta;
    std::vector<std::vector<std::vector<double> > > expected_dDirectionCosinesdGamma;
    double                                          frac = 0.70710678118654757;
    double                                          half = 0.5;

    bungeEulerAngles = {
        {M_PI,   0.,     0.    },
        {0.,     0.,     M_PI  },
        {0.,     M_PI,   0.    },
        {M_PI,   M_PI,   0.    },
        {M_PI,   M_PI_2, 0.    },
        {0.,     M_PI_2, M_PI  },
        {M_PI_4, M_PI_4, 0.    },
        {0.,     M_PI_4, M_PI_4},
    };
    expectedDirectionCosines = {
        {{-1., 0., 0.},      {0., -1., 0.},     {0., 0., 1.}     },
        {{-1., 0., 0.},      {0., -1., 0.},     {0., 0., 1.}     },
        {{1., 0., 0.},       {0., -1., 0.},     {0., 0., -1.}    },
        {{-1., 0., 0.},      {0., 1., 0.},      {0., 0., -1.}    },
        {{-1., 0., 0.},      {0., 0., 1.},      {0., 1., 0.}     },
        {{-1., 0., 0.},      {0., 0., -1.},     {0., -1., 0.}    },
        {{frac, -0.5, 0.5},  {frac, 0.5, -0.5}, {0.0, frac, frac}},
        {{frac, -frac, 0.0}, {0.5, 0.5, -frac}, {0.5, 0.5, frac} },
    };
    expected_dDirectionCosinesdAlpha = {
        {{0., 1., 0.},         {-1., 0., 0.},       {0., 0., 0.}},
        {{0., 1., 0.},         {-1., 0., 0.},       {0., 0., 0.}},
        {{0., 1., 0.},         {1., 0., 0.},        {0., 0., 0.}},
        {{0., -1., 0.},        {-1., 0., 0.},       {0., 0., 0.}},
        {{0., 0., -1.},        {-1., 0., 0.},       {0., 0., 0.}},
        {{0., 0., 1.},         {-1., 0., 0.},       {0., 0., 0.}},
        {{-frac, -half, half}, {frac, -half, half}, {0., 0., 0.}},
        {{-half, -half, frac}, {frac, -frac, 0.},   {0., 0., 0.}},
    };
    expected_dDirectionCosinesdBeta = {
        {{0., 0., 0.},     {0., 0., 1.},          {0., 1., 0.}       },
        {{0., 0., 0.},     {0., 0., -1.},         {0., -1., 0.}      },
        {{0., 0., 0.},     {0., 0., 1.},          {0., -1., 0.}      },
        {{0., 0., 0.},     {0., 0., -1.},         {0., -1., 0.}      },
        {{0., 0., 0.},     {0., 1., 0.},          {0., 0., -1.}      },
        {{0., 0., 0.},     {0., 1., 0.},          {0., 0., -1.}      },
        {{0., half, half}, {0., -half, -half},    {0., frac, -frac}  },
        {{0., 0., 0.},     {-half, -half, -frac}, {half, half, -frac}},
    };
    expected_dDirectionCosinesdGamma = {
        {{0., 1., 0.},       {-1., 0., 0.},     {0., 0., 0.}     },
        {{0., 1., 0.},       {-1., 0., 0.},     {0., 0., 0.}     },
        {{0., -1., 0.},      {-1., 0., 0.},     {0., 0., 0.}     },
        {{0., 1., 0.},       {1., 0., 0.},      {0., 0., 0.}     },
        {{0., 1., 0.},       {0., 0., 0.},      {1., 0., 0.}     },
        {{0., 1., 0.},       {0., 0., 0.},      {-1., 0., 0.}    },
        {{-half, -frac, 0.}, {half, -frac, 0.}, {frac, 0., 0.}   },
        {{-frac, -frac, 0.}, {half, -half, 0.}, {half, -half, 0.}},
    };

    for (unsigned int i = 0; i < bungeEulerAngles.size(); i++) {
        // Matrix directionCosines interface
        tardigradeVectorTools::rotationMatrix(bungeEulerAngles[i], directionCosines);
        BOOST_TEST(tardigradeVectorTools::appendVectors(directionCosines) ==
                       tardigradeVectorTools::appendVectors(expectedDirectionCosines[i]),
                   boost::test_tools::per_element());
        // Row-major vector interface
        tardigradeVectorTools::rotationMatrix(bungeEulerAngles[i], directionCosinesVector);
        BOOST_TEST(directionCosinesVector == tardigradeVectorTools::appendVectors(expectedDirectionCosines[i]),
                   boost::test_tools::per_element());
        // Matrix directionCosines and partial derivatives interface
        tardigradeVectorTools::rotationMatrix(bungeEulerAngles[i], directionCosines, dDirectionCosinesdAlpha,
                                              dDirectionCosinesdBeta, dDirectionCosinesdGamma);
        BOOST_TEST(tardigradeVectorTools::appendVectors(directionCosines) ==
                       tardigradeVectorTools::appendVectors(expectedDirectionCosines[i]),
                   boost::test_tools::per_element());
        BOOST_TEST(tardigradeVectorTools::appendVectors(dDirectionCosinesdAlpha) ==
                       tardigradeVectorTools::appendVectors(expected_dDirectionCosinesdAlpha[i]),
                   boost::test_tools::per_element());
        BOOST_TEST(tardigradeVectorTools::appendVectors(dDirectionCosinesdBeta) ==
                       tardigradeVectorTools::appendVectors(expected_dDirectionCosinesdBeta[i]),
                   boost::test_tools::per_element());
        BOOST_TEST(tardigradeVectorTools::appendVectors(dDirectionCosinesdGamma) ==
                       tardigradeVectorTools::appendVectors(expected_dDirectionCosinesdGamma[i]),
                   boost::test_tools::per_element());
    }
}

BOOST_AUTO_TEST_CASE(test_solveLinearSystem, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the utility to solve a linear system of equations.
     */

    matrixType A = {
        {0.95617934, 0.41311152, 0.25812163},
        {0.13346546, 0.12864080, 0.65152997},
        {0.15425579, 0.35444563, 0.43177476}
    };

    vectorType b = {0.92263585, 0.96720703, 0.72477723};

    vectorType xSolution = {0.52372037, 0.18327279, 1.34104665};

    unsigned int rank;

    vectorType xAnswer = tardigradeVectorTools::solveLinearSystem(A, b, rank);

    BOOST_TEST(xSolution == xAnswer, CHECK_PER_ELEMENT);

    BOOST_TEST(rank == 3);

    tardigradeVectorTools::solverType<floatType> linearSolver;
    xAnswer = tardigradeVectorTools::solveLinearSystem(A, b, rank, linearSolver);

    BOOST_TEST(xSolution == xAnswer, CHECK_PER_ELEMENT);

    BOOST_CHECK(rank == 3);

    xAnswer = vectorType(3, 0);
    Eigen::Map<Eigen::MatrixXd>       xmat(xAnswer.data(), 3, 1);
    Eigen::Map<const Eigen::MatrixXd> bmat(b.data(), 3, 1);
    xmat = linearSolver.solve(bmat);

    BOOST_TEST(xSolution == xAnswer, CHECK_PER_ELEMENT);

    A = {{2}};
    b = {7};

    xSolution = {3.5};

    xAnswer = tardigradeVectorTools::solveLinearSystem(A, b, rank);

    BOOST_TEST(xAnswer == xSolution, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_isParallel, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the utility that tests if two vectors are parallel or not
     */

    vectorType v1 = {1, 2, 3};
    vectorType v2 = {2, 4, 6};

    BOOST_CHECK(tardigradeVectorTools::isParallel(v1, v2));

    v1 = {0, 2, 3};
    BOOST_CHECK(!tardigradeVectorTools::isParallel(v1, v2));

    std::vector<int> v3 = {1, 2, 3};
    std::vector<int> v4 = {2, 4, 6};

    BOOST_CHECK(tardigradeVectorTools::isParallel(v3, v4));

    v3[0] = 0;
    BOOST_CHECK(!tardigradeVectorTools::isParallel(v3, v4));
}

// TODO: Parametrize with BOOST_DATA_TEST_CASE
BOOST_AUTO_TEST_CASE(test_isOrthogonal_verifyOrthogonal, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the utility that tests if two vectors are orthogonal or not
     */

    std::vector<std::vector<int> > v1 = {
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 0, 0},
        {1, 1, 0},
        {1, 1, 0},
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    // Orthogonal with v1
    std::vector<std::vector<int> > v2 = {
        {0,  1,  0 },
        {0,  0,  1 },
        {0,  2,  0 },
        {0,  1,  1 },
        {0,  2,  2 },
        {0,  0,  1 },
        {0,  0,  2 },
        {1,  -1, 0 },
        {-1, 0,  1 },
        {0,  1,  -1},
    };
    // Not orthogonal with v1
    std::vector<std::vector<int> > v3 = {
        {1,  1,  0 },
        {1,  0,  1 },
        {1,  2,  0 },
        {1,  1,  1 },
        {1,  2,  2 },
        {0,  1,  1 },
        {0,  1,  2 },
        {1,  -1, 1 },
        {-1, 1,  1 },
        {1,  1,  -1},
    };
    std::vector<double> v1Float;
    std::vector<double> v2Float;
    std::vector<double> v3Float;  // Not orthogonal with v1

    for (unsigned int iTest = 0; iTest < v1.size(); iTest++) {
        // Not orthogonal. Return False or throw exception
        BOOST_CHECK(!tardigradeVectorTools::isOrthogonal(v1[iTest], v1[iTest]));
        BOOST_CHECK(!tardigradeVectorTools::isOrthogonal(v1[iTest], v3[iTest]));
        BOOST_CHECK_THROW(tardigradeVectorTools::verifyOrthogonal(v1[iTest], v1[iTest]), std::runtime_error);
        BOOST_CHECK_THROW(tardigradeVectorTools::verifyOrthogonal(v1[iTest], v3[iTest]), std::runtime_error);

        // Orthogonal. Return True or no throw.
        BOOST_CHECK(tardigradeVectorTools::isOrthogonal(v1[iTest], v2[iTest]));
        BOOST_CHECK_NO_THROW(tardigradeVectorTools::verifyOrthogonal(v1[iTest], v2[iTest]));

        // Cast to float for type template feature check
        std::copy(v1[iTest].begin(), v1[iTest].end(), back_inserter(v1Float));
        std::copy(v2[iTest].begin(), v2[iTest].end(), back_inserter(v2Float));
        std::copy(v3[iTest].begin(), v3[iTest].end(), back_inserter(v3Float));

        // Not orthogonal. Return False or throw exception
        BOOST_CHECK(!tardigradeVectorTools::isOrthogonal(v1Float, v1Float));
        BOOST_CHECK(!tardigradeVectorTools::isOrthogonal(v1Float, v3Float));
        BOOST_CHECK_THROW(tardigradeVectorTools::verifyOrthogonal(v1Float, v1Float), std::runtime_error);
        BOOST_CHECK_THROW(tardigradeVectorTools::verifyOrthogonal(v1Float, v3Float), std::runtime_error);

        // Orthogonal. Return True or no throw.
        BOOST_CHECK(tardigradeVectorTools::isOrthogonal(v1Float, v2Float));
        BOOST_CHECK_NO_THROW(tardigradeVectorTools::verifyOrthogonal(v1Float, v2Float));
    }
}

BOOST_AUTO_TEST_CASE(test_dyadic, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the dyadic product between two vectors
     */

    vectorType v1 = {1, 2, 3};
    vectorType v2 = {4, 5, 6};
    matrixType A  = tardigradeVectorTools::dyadic(v1, v2);

    vectorType answer = {4, 5, 6, 8, 10, 12, 12, 15, 18};

    BOOST_TEST(tardigradeVectorTools::appendVectors(A) == answer, CHECK_PER_ELEMENT);

    vectorType flat_A(9, 0);

    tardigradeVectorTools::rowMajorDyadic<floatType>(std::begin(v1), std::end(v1), std::begin(v2), std::end(v2),
                                                     std::begin(flat_A), std::end(flat_A));

    BOOST_TEST(flat_A == answer, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_eye, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the formation of an identity matrix
     */

    std::vector<double> Ivec(9, 1.);
    std::vector<double> IvecExpected = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    tardigradeVectorTools::eye(Ivec);
    BOOST_TEST(Ivec == IvecExpected, CHECK_PER_ELEMENT);

    unsigned int                      dim = 4;
    std::vector<std::vector<double> > I   = tardigradeVectorTools::eye<double>(dim);

    std::vector<double> I_answer = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    BOOST_TEST(tardigradeVectorTools::appendVectors(I) == I_answer, CHECK_PER_ELEMENT);

    I.clear();
    tardigradeVectorTools::eye(dim, I);

    BOOST_TEST(tardigradeVectorTools::appendVectors(I) == I_answer, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_determinant, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the determinant of a matrix
     */

    std::vector<floatType> Avec = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    BOOST_TEST(tardigradeVectorTools::determinant(Avec, 3, 3) == 1.);

    Avec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    BOOST_TEST(tardigradeVectorTools::determinant(Avec, 3, 3) == 0.);

    Avec = {1, 2, 3, 4};
    BOOST_TEST(tardigradeVectorTools::determinant(Avec, 2, 2) == -2.);
}

BOOST_AUTO_TEST_CASE(test_inverse, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the matrix inverse;
     */

    std::vector<floatType> Avec = {0.39874077, 0.11561812, -0.75485222, 0.14034205, 0.15851022,
                                   1.29640525, 0.26235075, -0.26051883, 0.45378251};

    std::vector<floatType> answer = {1.6109566,   0.56699734,  1.0599259,  1.08701345, 1.49027454,
                                     -2.44933462, -0.30730183, 0.52776914, 0.18473576};

    std::vector<floatType> Avecinv = tardigradeVectorTools::inverse(Avec, 3, 3);

    BOOST_TEST(Avecinv == answer, CHECK_PER_ELEMENT);

    std::vector<std::vector<floatType> > A    = tardigradeVectorTools::inflate(Avec, 3, 3);
    std::vector<std::vector<double> >    Ainv = tardigradeVectorTools::inverse(A);

    BOOST_TEST(tardigradeVectorTools::appendVectors(Ainv) == answer, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_inflate, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the inflate command in tardigrade_vector_tools
     */

    std::vector<floatType> Avec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    unsigned int nrows = 2;
    unsigned int ncols = 5;

    std::vector<std::vector<floatType> > answer = {
        {1, 2, 3, 4, 5 },
        {6, 7, 8, 9, 10}
    };

    std::vector<std::vector<floatType> > result = tardigradeVectorTools::inflate(Avec, nrows, ncols);

    for (unsigned int i = 0; i < nrows; i++) {
        BOOST_TEST(answer[i] == result[i], CHECK_PER_ELEMENT);
    }
}

BOOST_AUTO_TEST_CASE(test_computeDDetADA, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the derivative of the determinant w.r.t.
     * the matrix.
     */

    std::vector<floatType> A = {0.39874077, 0.11561812, -0.75485222, 0.14034205, 0.15851022,
                                1.29640525, 0.26235075, -0.26051883, 0.45378251};

    std::vector<floatType> ddetAdA = tardigradeVectorTools::computeDDetADA(A, 3, 3);

    floatType eps   = 1e-6;
    floatType detA0 = tardigradeVectorTools::determinant(A, 3, 3);
    floatType detA;

    for (unsigned int i = 0; i < A.size(); i++) {
        std::vector<floatType> delta(A.size(), 0);
        delta[i] = fabs(A[i] * eps);
        detA     = tardigradeVectorTools::determinant(A + delta, 3, 3);

        BOOST_TEST((detA - detA0) / delta[i] == ddetAdA[i]);
    }
}

BOOST_AUTO_TEST_CASE(test_matrixMultiply, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the matrix multiplication function.
     */

    vectorType A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    vectorType B = {10, 11, 12, 13, 14, 15, 16, 17, 18};
    vectorType C;

    vectorType answer_1 = {84, 90, 96, 201, 216, 231, 318, 342, 366};
    vectorType answer_2 = {174, 186, 198, 213, 228, 243, 252, 270, 288};
    vectorType answer_3 = {68, 86, 104, 167, 212, 257, 266, 338, 410};
    vectorType answer_4 = {138, 174, 210, 171, 216, 261, 204, 258, 312};

    C = tardigradeVectorTools::matrixMultiply(A, B, 3, 3, 3, 3, 0, 0);

    BOOST_TEST(C == answer_1, CHECK_PER_ELEMENT);

    C = tardigradeVectorTools::matrixMultiply(A, B, 3, 3, 3, 3, 1, 0);

    BOOST_TEST(C == answer_2, CHECK_PER_ELEMENT);

    C = tardigradeVectorTools::matrixMultiply(A, B, 3, 3, 3, 3, 0, 1);

    BOOST_TEST(C == answer_3, CHECK_PER_ELEMENT);

    C = tardigradeVectorTools::matrixMultiply(A, B, 3, 3, 3, 3, 1, 1);

    BOOST_TEST(C == answer_4, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_matrixSqrtResidual, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the residual used in solving for
     * the square root of a matrix.
     */

    vectorType A = {3., 3., 5., 3., 7., 7., 5., 7., 11.};
    vectorType X = {1., 2., 3., 4., 5., 6., 7., 8., 9.};

    vectorType R(9), RJp(9), RJm(9);
    vectorType J(81), JJ(81);

    tardigradeVectorTools::__matrixSqrtResidual(std::begin(A), std::end(A), 3, std::begin(X), std::end(X),
                                                std::begin(R), std::end(R), std::begin(J), std::end(J));

    // Check that the Jacobian is consistent with the residual
    floatType eps = 1e-6;
    for (unsigned int i = 0; i < X.size(); i++) {
        vectorType delta(X.size(), 0);
        delta[i] = eps * fabs(X[i]) + eps;

        vectorType Xp = X + delta;
        vectorType Xm = X - delta;

        tardigradeVectorTools::__matrixSqrtResidual(std::begin(A), std::end(A), 3, std::begin(Xp), std::end(Xp),
                                                    std::begin(RJp), std::end(RJp), std::begin(JJ), std::end(JJ));
        tardigradeVectorTools::__matrixSqrtResidual(std::begin(A), std::end(A), 3, std::begin(Xm), std::end(Xm),
                                                    std::begin(RJm), std::end(RJm), std::begin(JJ), std::end(JJ));

        vectorType gradCol = (RJp - RJm) / (2 * delta[i]);

        for (unsigned int j = 0; j < A.size(); j++) {
            BOOST_TEST(gradCol[j] == J[9 * j + i]);
        }
    }
}

BOOST_AUTO_TEST_CASE(test_matrixSqrt, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the square root of a matrix.
     */

    vectorType   A   = {3., 3., 5., 3., 7., 7., 5., 7., 11.};
    unsigned int dim = 3;
    floatType    eps = 1e-6;
    vectorType   gradCol;

    vectorType X = tardigradeVectorTools::matrixSqrt(A, 3);

    BOOST_TEST(A == tardigradeVectorTools::matrixMultiply(X, X, dim, dim, dim, dim, 0, 0), CHECK_PER_ELEMENT);

    // Test the jacobian
    matrixType dAdX;
    vectorType XJ = tardigradeVectorTools::matrixSqrt(A, 3, dAdX);

    BOOST_TEST(XJ == X, CHECK_PER_ELEMENT);

    vectorType dXdA = tardigradeVectorTools::inverse(tardigradeVectorTools::appendVectors(dAdX), A.size(), A.size());

    for (unsigned int i = 0; i < A.size(); i++) {
        vectorType delta(A.size(), 0);
        delta[i] = eps * A[i] + eps;

        vectorType XJp = tardigradeVectorTools::matrixSqrt(A + delta, 3);
        vectorType XJm = tardigradeVectorTools::matrixSqrt(A - delta, 3);

        gradCol = (XJp - XJm) / (2 * delta[i]);

        for (unsigned int j = 0; j < gradCol.size(); j++) {
            BOOST_TEST(gradCol[j] == dXdA[A.size() * j + i]);
        }
    }
}

BOOST_AUTO_TEST_CASE(test_median, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the median of a vector
     */

    vectorType x = {0.65353053, 0.97839806, 0.32387778, 0.13137077, 0.17253149, 0.03216338};
    BOOST_TEST(tardigradeVectorTools::median(x) == 0.24820463352966032);

    x = {0.82387829, 0.07615528, 0.89366009, 0.04843728, 0.81331188, 0.19575555, 0.02308312};
    BOOST_TEST(tardigradeVectorTools::median(x) == .19575554955487007);
}

BOOST_AUTO_TEST_CASE(test_abs, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the absolute value of a vector.
     */

    std::vector<double> x        = {-1, 2, 3, 4, -5, 6};
    std::vector<double> x_answer = {1, 2, 3, 4, 5, 6};

    BOOST_TEST(tardigradeVectorTools::abs(x) == x_answer, CHECK_PER_ELEMENT);

    std::vector<int> y        = {-1, 2, 3, 4, -5, 6};
    std::vector<int> y_answer = {1, 2, 3, 4, 5, 6};

    BOOST_TEST(tardigradeVectorTools::abs(y) == y_answer, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_svd, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the singular value decomposition of a matrix.
     */

    const vectorType A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    vectorType UAnswer1(9, 0);
    tardigradeVectorTools::eye(UAnswer1);

    vectorType SigmaAnswer1 = {2.54368356e+01, 1.72261225e+00, 2.64839734e-16};

    vectorType VAnswer1(16, 0);
    tardigradeVectorTools::eye(VAnswer1);

    vectorType UAnswer2(16, 0);
    tardigradeVectorTools::eye(UAnswer2);

    vectorType SigmaAnswer2 = {2.54624074e+01, 1.29066168e+00, 1.38648772e-15};

    vectorType VAnswer2(9, 0);
    tardigradeVectorTools::eye(VAnswer2);

    vectorType UResult;
    vectorType SigmaResult;
    vectorType VResult;

    // Test the first orientation of A

    tardigradeVectorTools::svd(A, 3, 4, UResult, SigmaResult, VResult);

    // Check that the singular values are correct

    BOOST_CHECK(tardigradeVectorTools::fuzzyEquals(
        SigmaAnswer1, SigmaResult));  // Leaving as fuzzy equals because the final value is noise

    // Check that the left and right singular values are orthogonal

    BOOST_TEST(UAnswer1 == tardigradeVectorTools::matrixMultiply(UResult, UResult, 3, 3, 3, 3, 0, 1),
               CHECK_PER_ELEMENT);

    BOOST_TEST(VAnswer1 == tardigradeVectorTools::matrixMultiply(VResult, VResult, 4, 4, 4, 4, 0, 1),
               CHECK_PER_ELEMENT);

    // Test the second orientation of A

    tardigradeVectorTools::svd(A, 4, 3, UResult, SigmaResult, VResult);

    // Check that the singular values are correct

    BOOST_CHECK(tardigradeVectorTools::fuzzyEquals(
        SigmaAnswer2, SigmaResult));  // Leaving as fuzzy equals because the final value is noise

    // Check that the left and right singular values are orthogonal

    BOOST_TEST(UAnswer2 == tardigradeVectorTools::matrixMultiply(UResult, UResult, 4, 4, 4, 4, 0, 1),
               CHECK_PER_ELEMENT);

    BOOST_TEST(VAnswer2 == tardigradeVectorTools::matrixMultiply(VResult, VResult, 3, 3, 3, 3, 0, 1),
               CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_polar_decomposition, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the polar decomposition function
     */

    vectorType A = {2, 2, 3, 4, 6, 6, 7, 8, 10};

    vectorType UAnswer = {4.10159295, 4.41090403, 5.72021511, 4.41090403, 6.47138062,
                          6.5318572,  5.72021511, 6.5318572,  8.3434993};

    vectorType VAnswer = {1.19923598, 1.8434344,  3.48763282, 1.8434344,  5.5783477,
                          7.31326101, 3.48763282, 7.31326101, 12.13888919};

    vectorType RAnswer = {-0.41938618, -0.27751878, 0.86434863, -0.27751878, 0.94573945,
                          0.16899768,  0.86434863,  0.16899768, 0.47364673};

    vectorType UResult;

    vectorType VResult;

    vectorType RResult;

    // Test the right polar decomposition

    tardigradeVectorTools::polar_decomposition(A, 3, 3, RResult, UResult, false);

    BOOST_TEST(UResult == UAnswer, CHECK_PER_ELEMENT);

    BOOST_TEST(RResult == RAnswer, CHECK_PER_ELEMENT);

    // Test the left polar decomposition

    tardigradeVectorTools::polar_decomposition(A, 3, 3, RResult, VResult, true);

    BOOST_TEST(VResult == VAnswer, CHECK_PER_ELEMENT);

    BOOST_TEST(RResult == RAnswer, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_computeDinvAdA, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the derivative of the inverse of A w.r.t. A
     */

    vectorType A = {0.39293837,  -0.42772133, -0.54629709, 0.10262954, 0.43893794,
                    -0.15378708, 0.9615284,   0.36965948,  -0.0381362};

    vectorType invA = tardigradeVectorTools::inverse(A, 3, 3);

    floatType eps = 1e-6;

    matrixType gradient(A.size(), vectorType(A.size(), 0));

    for (unsigned int i = 0; i < A.size(); i++) {
        vectorType delta(A.size(), 0);

        delta[i] = eps * std::fabs(A[i]) + eps;

        vectorType invAp, invAm;

        BOOST_CHECK_NO_THROW(invAp = tardigradeVectorTools::inverse(A + delta, 3, 3));

        BOOST_CHECK_NO_THROW(invAm = tardigradeVectorTools::inverse(A - delta, 3, 3));

        for (unsigned int j = 0; j < A.size(); j++) {
            gradient[j][i] = (invAp[j] - invAm[j]) / (2 * delta[i]);
        }
    }

    BOOST_TEST(tardigradeVectorTools::appendVectors(gradient) ==
                   tardigradeVectorTools::appendVectors(tardigradeVectorTools::computeDInvADA(invA, 3, 3)),
               CHECK_PER_ELEMENT);

    vectorType dInvAdA(81, 0);
    tardigradeVectorTools::computeFlatDInvADA<3, 3>(std::cbegin(invA), std::cend(invA), std::begin(dInvAdA),
                                                    std::end(dInvAdA));

    BOOST_TEST(tardigradeVectorTools::appendVectors(gradient) == dInvAdA, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_computeMatrixExponential, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the derivative of the inverse of A w.r.t. A
     */

    vectorType A = {-0.39293837, 0.42772133, 0.54629709,  -0.10262954, -0.43893794,
                    0.15378708,  -0.9615284, -0.36965948, 0.0381362};

    unsigned int dim = 3;

    vectorType answer = {0.46127063, 0.17591871,  0.43827842, -0.11397388, 0.60563844,
                         0.0935307,  -0.71521989, -0.4244122, 0.78340236};

    vectorType result, resultJ;

    vectorType dExpAdA;

    tardigradeVectorTools::computeMatrixExponential(A, dim, result);

    BOOST_TEST(answer == result, CHECK_PER_ELEMENT);

    tardigradeVectorTools::computeMatrixExponential(A, dim, resultJ, dExpAdA);

    BOOST_TEST(answer == resultJ, CHECK_PER_ELEMENT);

    vectorType J(dim * dim * dim * dim, 0);

    floatType eps = 1e-6;

    for (unsigned int i = 0; i < A.size(); i++) {
        vectorType delta(A.size(), 0);

        delta[i] = eps * std::fabs(A[i]) + eps;

        vectorType expAp, expAm;

        BOOST_CHECK_NO_THROW(tardigradeVectorTools::computeMatrixExponential(A + delta, 3, expAp));

        BOOST_CHECK_NO_THROW(tardigradeVectorTools::computeMatrixExponential(A - delta, 3, expAm));

        for (unsigned int j = 0; j < A.size(); j++) {
            J[9 * j + i] = (expAp[j] - expAm[j]) / (2 * delta[i]);
        }
    }

    BOOST_TEST(J == dExpAdA, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_computeMatrixExponential2, *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the derivative of the inverse of A w.r.t. A
     */

    vectorType A = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    unsigned int dim = 3;

    vectorType answer = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    vectorType result, resultJ;

    vectorType dExpAdA;

    tardigradeVectorTools::computeMatrixExponential(A, dim, result);

    BOOST_TEST(answer == result, CHECK_PER_ELEMENT);

    tardigradeVectorTools::computeMatrixExponential(A, dim, resultJ, dExpAdA);

    BOOST_TEST(answer == resultJ, CHECK_PER_ELEMENT);

    vectorType J(dim * dim * dim * dim, 0);

    floatType eps = 1e-6;

    for (unsigned int i = 0; i < A.size(); i++) {
        vectorType delta(A.size(), 0);

        delta[i] = eps * std::fabs(A[i]) + eps;

        vectorType expAp, expAm;

        BOOST_CHECK_NO_THROW(tardigradeVectorTools::computeMatrixExponential(A + delta, 3, expAp));

        BOOST_CHECK_NO_THROW(tardigradeVectorTools::computeMatrixExponential(A - delta, 3, expAm));

        for (unsigned int j = 0; j < A.size(); j++) {
            J[9 * j + i] = (expAp[j] - expAm[j]) / (2 * delta[i]);
        }
    }

    BOOST_TEST(J == dExpAdA, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_computeMatrixExponentialScalingAndSquaring,
                     *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the derivative of the inverse of A w.r.t. A
     */

    vectorType A = {-0.39293837, 0.42772133, 0.54629709,  -0.10262954, -0.43893794,
                    0.15378708,  -0.9615284, -0.36965948, 0.0381362};

    unsigned int dim = 3;

    vectorType answer = {0.46127063, 0.17591871,  0.43827842, -0.11397388, 0.60563844,
                         0.0935307,  -0.71521989, -0.4244122, 0.78340236};

    vectorType result, resultJ;

    vectorType dExpAdA;

    tardigradeVectorTools::computeMatrixExponentialScalingAndSquaring(A, dim, result);

    BOOST_TEST(answer == result, CHECK_PER_ELEMENT);

    tardigradeVectorTools::computeMatrixExponentialScalingAndSquaring(A, dim, resultJ, dExpAdA);

    BOOST_TEST(answer == resultJ, CHECK_PER_ELEMENT);

    vectorType J(dim * dim * dim * dim, 0);

    floatType eps = 1e-6;

    for (unsigned int i = 0; i < A.size(); i++) {
        vectorType delta(A.size(), 0);

        delta[i] = eps * std::fabs(A[i]) + eps;

        vectorType expAp, expAm;

        BOOST_CHECK_NO_THROW(tardigradeVectorTools::computeMatrixExponentialScalingAndSquaring(A + delta, 3, expAp));

        BOOST_CHECK_NO_THROW(tardigradeVectorTools::computeMatrixExponentialScalingAndSquaring(A - delta, 3, expAm));

        for (unsigned int j = 0; j < A.size(); j++) {
            J[9 * j + i] = (expAp[j] - expAm[j]) / (2 * delta[i]);
        }
    }

    BOOST_TEST(J == dExpAdA, CHECK_PER_ELEMENT);
}

BOOST_AUTO_TEST_CASE(test_computeMatrixExponentialScalingAndSquaring2,
                     *boost::unit_test::tolerance(DEFAULT_TEST_TOLERANCE)) {
    /*!
     * Test the computation of the derivative of the inverse of A w.r.t. A
     */

    vectorType A = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    unsigned int dim = 3;

    vectorType answer = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    vectorType result, resultJ;

    vectorType dExpAdA;

    tardigradeVectorTools::computeMatrixExponentialScalingAndSquaring(A, dim, result);

    BOOST_TEST(answer == result, CHECK_PER_ELEMENT);

    tardigradeVectorTools::computeMatrixExponentialScalingAndSquaring(A, dim, resultJ, dExpAdA);

    BOOST_TEST(answer == resultJ, CHECK_PER_ELEMENT);

    vectorType J(dim * dim * dim * dim, 0);

    floatType eps = 1e-6;

    for (unsigned int i = 0; i < A.size(); i++) {
        vectorType delta(A.size(), 0);

        delta[i] = eps * std::fabs(A[i]) + eps;

        vectorType expAp, expAm;

        BOOST_CHECK_NO_THROW(tardigradeVectorTools::computeMatrixExponentialScalingAndSquaring(A + delta, 3, expAp));

        BOOST_CHECK_NO_THROW(tardigradeVectorTools::computeMatrixExponentialScalingAndSquaring(A - delta, 3, expAm));

        for (unsigned int j = 0; j < A.size(); j++) {
            J[9 * j + i] = (expAp[j] - expAm[j]) / (2 * delta[i]);
        }
    }

    BOOST_TEST(J == dExpAdA, CHECK_PER_ELEMENT);
}
