#ifndef _STATISTICSAGGREGTORS_H_
#define _STATISTICSAGGREGTORS_H_

// This file defines some IStatisticsAggregator implementations used by the
// example code in Classification.h, DensityEstimation.h, etc. Note we
// represent IStatisticsAggregator instances using simple structs so that all
// tree data can be stored contiguously in a linear array.

#include "include.h"
#include "DataPointCollection.h"

namespace Teutoburg
{
    class HistogramAggregator: public sw::IStatisticsAggregator<HistogramAggregator>
    {
    private:
        bp::dict bins;
        int sampleCount;
    public:
        HistogramAggregator();

        bp::dict GetPyObject(void);

        bp::tuple GetResponse(bp::object inp) const;

        inline void countUp(bp::object label, int step=1);

        int getSampleCount(void ) const;

        void Clear();

        void Aggregate(const sw::IDataPointCollection& data, unsigned int index);

        void Aggregate(const HistogramAggregator& aggregator);

        double Entropy() const;

        HistogramAggregator DeepClone() const;
    };

    /* Gaussian Aggregator
    we assume a linear homoscedastic model with multivariate Gaussian noise

    the linear prediction is determined by:
        estimating the mean/offset: b_est
        solving the least squares problem: y_train-b_est = x_train M_est for M_est
        y_train_est = x_train M_est + b_est
        y_test_est = x_test M_est + b_est
    the training residuals are calculated by:
        r = y_train - y_train_est
    the covariance of the residuals can be calculated: c_r
    the entropy is calculated on the residual covariance: E = 0.5*log(det(2*pi*e*c_r))


    => WRITE AGGREGATOR THAT COLLECTS ALL DATAPOINTS & LABELS INTO A LIST
    */
    class GaussianAggregator: public sw::IStatisticsAggregator<GaussianAggregator>
    {
    private:
        unsigned int sampleCount;
        unsigned int data_dims;
        unsigned int label_dims;
        bool uptodate;
        bp::object M;
        bp::object ATA;
        bp::object ATy;
        bp::object yTy;
    public:
        GaussianAggregator(unsigned int data_dims=0, unsigned int label_dims=0);

        double Entropy() const;
        int getSampleCount(void ) const;

        void Clear();
        void Aggregate(const sw::IDataPointCollection& data, unsigned int index);
        void Aggregate(const GaussianAggregator& other);
        GaussianAggregator DeepClone() const;

        void update();
        bp::object GetPyObject(void) const;
        bp::tuple GetResponse(bp::object inp) const;
    };
}


#endif /* end of include guard: _STATISTICSAGGREGTORS_H_ */
