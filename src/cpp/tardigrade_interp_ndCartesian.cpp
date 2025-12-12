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

#include <tardigrade_error_tools.h>

namespace tardigradeVectorTools {

    namespace interp {

        ndCartesian::ndCartesian(const unsigned int spatial_dimension, const floatType *const D,
                                 const unsigned int D_size, const unsigned int npts, const floatType tolr,
                                 const floatType tola)
            : _spatial_dimension(spatial_dimension),
              _D(D),
              _D_size(D_size),
              _npts(npts),
              _D_cols(_D_size / _npts),
              _tolr(tolr),
              _tola(tola) {
            /*!
             * The constructor function for ndCartesian
             */

            TARDIGRADE_ERROR_TOOLS_CHECK(_D_cols * _npts == _D_size,
                                         "The size of D ( " + std::to_string(_D_size) +
                                             " ) is not a scalar multiple of the number of points ( " +
                                             std::to_string(_npts) + " )");

            TARDIGRADE_ERROR_TOOLS_CHECK(_D_cols >= _spatial_dimension,
                                         "The spatial dimension ( " + std::to_string(_spatial_dimension) +
                                             " ) is larger than the number of columns in D ( " +
                                             std::to_string(_D_cols));

            // Set the dimensions of the cartesian grid
            setDimensions(getDimensions(_npts));

            // Set the strides for the dimensions
            setStrides();
        }

        std::vector<unsigned int> ndCartesian::getDimensions(const unsigned int span, const unsigned int index) {
            /*!
             * Get the number of points in each of the spatial dimensions
             *
             * \param span: The number of points to search in D
             * \param index: The index of the spatial dimension to search
             */

            const floatType tol = _tolr * std::fabs(*(_D + index)) + _tola;

            unsigned int n_remainder = 0;

            for (unsigned int n = 0; n < span; n++) {
                if (std::fabs(*(_D + _D_cols * n + index) - *(_D + index)) > tol) {
                    break;
                }

                n_remainder++;
            }

            std::vector<unsigned int> nd = {span / n_remainder};

            if ((index + 1) < _spatial_dimension) {
                std::vector<unsigned int> nd_sub = getDimensions(n_remainder, index + 1);

                nd.insert(nd.end(), nd_sub.begin(), nd_sub.end());
            }

            return nd;
        }

        void ndCartesian::setStrides() {
            /*!
             * Set the strides for each of the cartesian dimensions
             */

            _strides = std::vector<unsigned int>(_dimensions.size(), 0);

            unsigned int ndim = _dimensions.size();

            for (unsigned int d = 0; d < ndim; d++) {
                _strides[d] = 1;

                for (unsigned int i = d + 1; i < ndim; i++) {
                    _strides[d] *= _dimensions[i];
                }
            }
        }

        std::array<unsigned int, 2> ndCartesian::getBounds(const unsigned int index, const floatType &pd,
                                                           const unsigned int dim_npts) {
            /*!
             * Get the bounding indices for the point in the given dimension
             *
             * \param index: The spatial index of the provided value
             * \param pd: The value to find the bounding box in the given dimension's index
             * \param dim_npts: The number of points in the given stride
             */

            unsigned int ub = 0;

            unsigned int lb = 0;

            if (*(_D + _D_cols * _strides[index] * ub + index) > pd) {
                return {lb, ub};
            }

            while ((ub + 1) < dim_npts) {
                ub++;

                if (*(_D + _D_cols * _strides[index] * ub + index) > pd) {
                    break;
                }

                lb++;
            }

            return {lb, ub};
        }

        std::vector<unsigned int> ndCartesian::getBoundingBoxIndices(const std::vector<floatType> &p,
                                                                     const unsigned int            insize,
                                                                     const unsigned int            index) {
            /*!
             * Get the bounding box indices for the given point
             *
             * \param &p: The incoming point
             * \param &insize: The length of D to be exploring
             * \param index: The index to be exploring
             */

            std::array<unsigned int, 2> bounds = getBounds(index, p[index], insize / _strides[index]);

            std::vector<unsigned int> bounding_box(bounds.begin(), bounds.end());

            if ((index + 1) < _spatial_dimension) {
                std::vector<unsigned int> sub_bounds = getBoundingBoxIndices(p, _strides[index], index + 1);

                bounding_box.insert(bounding_box.end(), sub_bounds.begin(), sub_bounds.end());
            }

            return bounding_box;
        }

        std::vector<floatType> ndCartesian::getWeights(const std::vector<floatType>    &p,
                                                       const std::vector<unsigned int> &current_bounds,
                                                       const unsigned int insize, const unsigned int index) {
            /*!
             * Get the weights for the given point
             *
             * \param &p: The incoming point
             * \param &current_bounds: The current bounds of the given point
             * \param &insize: The length of D to be exploring
             * \param index: The index to be exploring
             */

            floatType xd_lb = *(_D + _D_cols * _strides[index] * current_bounds[2 * index + 0] + index);
            floatType xd_ub = *(_D + _D_cols * _strides[index] * current_bounds[2 * index + 1] + index);

            floatType w_lb, w_ub;

            if (current_bounds[2 * index + 0] == current_bounds[2 * index + 1]) {
                w_lb = 0.5;

                w_ub = 0.5;

            } else {
                w_lb = 1 - (p[index] - xd_lb) / (xd_ub - xd_lb);
                w_ub = (p[index] - xd_lb) / (xd_ub - xd_lb);
            }

            std::vector<floatType> weights = {w_lb, w_ub};

            if ((index + 1) < _spatial_dimension) {
                std::vector<floatType> sub_weights = getWeights(p, current_bounds, _strides[index], index + 1);

                weights.insert(weights.end(), sub_weights.begin(), sub_weights.end());
            }

            return weights;
        }

        floatType ndCartesian::interpolateFunction(const std::vector<floatType>    &p,
                                                   const std::vector<unsigned int> &current_bounds,
                                                   const std::vector<floatType>    &current_weights,
                                                   const unsigned int col, const unsigned int index,
                                                   const unsigned int offset) {
            /*!
             * Interpolate the function at the given point
             *
             * \param &p: The incoming point
             * \param &current_bounds: The current bounds of the given point
             * \param &current_weights: The current weights of the given point
             * \param &col: The column of D after the points to interpolate
             * \param index: The index to be exploring
             * \param offset: The offset for the access to the D vector
             */

            const unsigned int ilb = _D_cols * _strides[index] * current_bounds[2 * index + 0] + offset;

            const unsigned int iub = _D_cols * _strides[index] * current_bounds[2 * index + 1] + offset;

            if ((index + 1) < _spatial_dimension) {
                return current_weights[2 * index + 0] *
                           interpolateFunction(p, current_bounds, current_weights, col, index + 1, ilb) +
                       current_weights[2 * index + 1] *
                           interpolateFunction(p, current_bounds, current_weights, col, index + 1, iub);

            } else {
                return current_weights[2 * index + 0] * (*(_D + ilb + _spatial_dimension + col)) +
                       current_weights[2 * index + 1] * (*(_D + iub + _spatial_dimension + col));
            }
        }

        floatType ndCartesian::eval(std::vector<floatType> &p, const unsigned int col) {
            /*!
             * Evaluate the interpolation at the provided point
             *
             * \param &p: The point to interpolate
             * \param col: The column of the cartesian grid after the provided spatial points to interpolate
             */

            std::vector<unsigned int> current_bounds = getBoundingBoxIndices(p, _npts);

            std::vector<floatType> current_weights = getWeights(p, current_bounds, _npts);

            return interpolateFunction(p, current_bounds, current_weights, col);
        }

    }  // namespace interp

}  // namespace tardigradeVectorTools
