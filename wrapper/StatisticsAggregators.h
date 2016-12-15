#pragma once

// This file defines some IStatisticsAggregator implementations used by the
// example code in Classification.h, DensityEstimation.h, etc. Note we
// represent IStatisticsAggregator instances using simple structs so that all
// tree data can be stored contiguously in a linear array.

#include <math.h>

#include <limits>
#include <vector>

#include "Sherwood.h"

#include "DataPointCollection.h"
#include <boost/numeric/ublas/vector.hpp>
#include <numpy/ndarrayobject.h>

namespace bp = boost::python;
namespace bu = boost::numeric::ublas;
namespace sw = MicrosoftResearch::Cambridge::Sherwood;

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
}
