#ifndef _TRAININGCONTEXTS_H_
#define _TRAININGCONTEXTS_H_
// This file defines types used to illustrate the use of the decision forest
// library in simple multi-class classification task (2D data points).

#include "StatisticsAggregators.h"
#include "FeatureResponseFunctions.h"
#include "DataPointCollection.h"
#include "include.h"

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

    template<class F>
    class RegressionTrainingContext : public sw::ITrainingContext<F,GaussianAggregator>     // where F:IFeatureResponse
    {
        private:
            int dim_data;
            int dim_labels;

        public:
            RegressionTrainingContext(int dim_data, int dim_labels);

        // Implementation of ITrainingContext
            F GetRandomFeature(sw::Random& random);

            GaussianAggregator GetStatisticsAggregator(void);

            double ComputeInformationGain(const GaussianAggregator& allStatistics, const GaussianAggregator& leftStatistics, const GaussianAggregator& rightStatistics);

            bool ShouldTerminate(const GaussianAggregator& parent, const GaussianAggregator& leftChild, const GaussianAggregator& rightChild, double gain);
    };

}

#endif /* end of include guard: _TRAININGCONTEXTS_H_ */
