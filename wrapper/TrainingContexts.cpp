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




    template <class F>
    double RegressionTrainingContext<F>::ComputeInformationGain(const GaussianAggregator& allStatistics, const GaussianAggregator& leftStatistics, const GaussianAggregator& rightStatistics)
    {
        double entropyBefore = allStatistics.Entropy();

        unsigned int nTotalSamples = leftStatistics.getSampleCount() + rightStatistics.getSampleCount();

        if (nTotalSamples <= 1)
                return 0.0;

        double entropyAfter = (leftStatistics.getSampleCount() * leftStatistics.Entropy() + rightStatistics.getSampleCount() * rightStatistics.Entropy()) / nTotalSamples;

        return entropyBefore - entropyAfter;
    }

    template <class F>
    RegressionTrainingContext<F>::RegressionTrainingContext(int dim_data, int dim_labels)
    {
        this->dim_data = dim_data;
        this->dim_labels = dim_labels;
    }

    template <class F>
    F RegressionTrainingContext<F>::GetRandomFeature(sw::Random& random)
    {
        return F(random, dim_data);
    }

    template <class F>
    GaussianAggregator RegressionTrainingContext<F>::GetStatisticsAggregator(void)
    {
        return GaussianAggregator(dim_labels);
    }

    template <class F>
    bool RegressionTrainingContext<F>::ShouldTerminate(const GaussianAggregator& parent, const GaussianAggregator& leftChild, const GaussianAggregator& rightChild, double gain)
    {
        return gain < 0.01;
    }

}
