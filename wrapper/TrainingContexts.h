#pragma once

// This file defines types used to illustrate the use of the decision forest
// library in simple multi-class classification task (2D data points).

#include <stdexcept>
#include <algorithm>

#include "Sherwood.h"

#include "StatisticsAggregators.h"
#include "FeatureResponseFunctions.h"
#include "DataPointCollection.h"

namespace sw = MicrosoftResearch::Cambridge::Sherwood;

namespace Teutoburg
{

template<class F>
class ClassificationTrainingContext : public sw::ITrainingContext<F,HistogramAggregator>     // where F:IFeatureResponse
{
    private:
        int dim;
        int nClasses;

    public:
        ClassificationTrainingContext(int dim, int nClasses);

    // Implementation of ITrainingContext
        F GetRandomFeature(sw::Random& random);

        HistogramAggregator GetStatisticsAggregator(void);

        double ComputeInformationGain(const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics);

        bool ShouldTerminate(const HistogramAggregator& parent, const HistogramAggregator& leftChild, const HistogramAggregator& rightChild, double gain);
    };

}
