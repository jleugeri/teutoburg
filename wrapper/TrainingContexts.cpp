#include "FeatureResponseFunctions.h"
#include "TrainingContexts.h"

namespace Teutoburg
{
    template <class F>
    double ClassificationTrainingContext<F>::ComputeInformationGain(const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics)
    {
        double entropyBefore = allStatistics.Entropy();

        unsigned int nTotalSamples = leftStatistics.getSampleCount() + rightStatistics.getSampleCount();

        if (nTotalSamples <= 1)
                return 0.0;

        double entropyAfter = (leftStatistics.getSampleCount() * leftStatistics.Entropy() + rightStatistics.getSampleCount() * rightStatistics.Entropy()) / nTotalSamples;

        return entropyBefore - entropyAfter;
    }

    template <class F>
    ClassificationTrainingContext<F>::ClassificationTrainingContext(int dim, int nClasses)
    {
        this->nClasses = nClasses;
        this->dim = dim;
    }

    template <class F>
    F ClassificationTrainingContext<F>::GetRandomFeature(sw::Random& random)
    {
        return F(random, dim);
    }

    template <class F>
    HistogramAggregator ClassificationTrainingContext<F>::GetStatisticsAggregator(void)
    {
        return HistogramAggregator(nClasses);
    }

    template <class F>
    bool ClassificationTrainingContext<F>::ShouldTerminate(const HistogramAggregator& parent, const HistogramAggregator& leftChild, const HistogramAggregator& rightChild, double gain)
    {
        return gain < 0.01;
    }

}
