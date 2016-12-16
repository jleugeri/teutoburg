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
        bp::object bins;
        int sampleCount;
        int nClasses;
    public:
        HistogramAggregator(int nClasses=0);

        bp::object GetPyObject(void);

        inline void countUp(int label, int step=1);

        int getSampleCount(void ) const;

        void Clear();

        void Aggregate(const sw::IDataPointCollection& data, unsigned int index);

        void Aggregate(const HistogramAggregator& aggregator);

        double Entropy() const;

        HistogramAggregator DeepClone() const;
    };

    class GaussianAggregator: public sw::IStatisticsAggregator<GaussianAggregator>
    {
    private:
        int sampleCount;
        int ndims;
        bp::object mean;
        bp::object squares;
    public:
        GaussianAggregator(int ndims=0);

        bp::object getMean(void) const;
        bp::object getCovariance(void) const;
        bp::object GetPyObject(void);

        int getSampleCount(void ) const;

        void Clear();

        void Aggregate(const sw::IDataPointCollection& data, unsigned int index);

        void Aggregate(const GaussianAggregator& aggregator);

        double Entropy() const;

        GaussianAggregator DeepClone() const;
    };
}


#endif /* end of include guard: _STATISTICSAGGREGTORS_H_ */
