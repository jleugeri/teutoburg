#include "StatisticsAggregators.h"
#include "DataPointCollection.h"

namespace Teutoburg
{
    bp::object HistogramAggregator::GetPyObject(void)
    {
        return bins.attr("__mul__")(1.0f/(double)sampleCount);
    }

  int HistogramAggregator::getSampleCount() const
  {
      return sampleCount;
  }

  // IStatisticsAggregator implementation
  void HistogramAggregator::Clear()
  {
      bins.attr("fill")(0);
      sampleCount = 0;
  }

  inline void HistogramAggregator::countUp(int label, int step)
  {
      bins.attr("__setitem__")(label, step+bp::extract<int>(bins.attr("__getitem__")(label).attr("item")()));
      sampleCount += step;
  }

  double HistogramAggregator::Entropy() const
  {
      double result = 0.0;

      for(int i=0; i<nClasses; ++i)
      {
          double val= (double) bp::extract<int>( bins.attr("__getitem__")(i).attr("item")());
          if(val!=0.0)
          {
              val /= (double)sampleCount;
              result -= val*log2(val);
          }

      }

      return result;
  }

  void HistogramAggregator::Aggregate(const sw::IDataPointCollection& data, unsigned int index)
  {
      const DataPointCollection& concreteData = (const DataPointCollection&)(data);
      int label = bp::extract<int>(concreteData.getLabelItem(index).attr("item")());

      countUp(label);
  }

  void HistogramAggregator::Aggregate(const HistogramAggregator& aggregator)
  {
      bins.attr("__iadd__")(aggregator.bins);
      sampleCount += aggregator.sampleCount;
  }

  HistogramAggregator HistogramAggregator::DeepClone() const
  {
      HistogramAggregator result(nClasses);
      result.Aggregate(*this);
      return result;
  }

  bp::object GaussianAggregator::getMean(void) const
  {
      return mean.attr("__mul__")(1.0f/(double)sampleCount);
  }

  bp::object GaussianAggregator::getCovariance(void) const
  {
      bp::object m = getMean();
      bp::object covariance = squares - np.attr("outer")(m,m);
      return covariance.attr("__mul__")(1.0f/(double)sampleCount);
  }

    bp::object GaussianAggregator::GetPyObject(void)
    {
        //make tuple of mean and covariance
        return bp::make_tuple(mean.attr("__mul__")(1.0f/(double)sampleCount), getCovariance());
    }

  int GaussianAggregator::getSampleCount() const
  {
      return sampleCount;
  }

  // IStatisticsAggregator implementation
  void GaussianAggregator::Clear()
  {
      mean.attr("fill")(0);
      squares.attr("fill")(0);
      sampleCount = 0;
  }

  double GaussianAggregator::Entropy() const
  {
      // Call numpy determinant function instead
      double det = bp::extract<double>(np.attr("linalg").attr("det")(getCovariance()));
      //return log(2*bp::numeric::pi*bp::numeric::e) + 0.5*log(det)
      return 2.8378770664093453 + 0.5*log(det);
  }

  void GaussianAggregator::Aggregate(const sw::IDataPointCollection& data, unsigned int index)
  {
      const DataPointCollection& concreteData = (const DataPointCollection&)(data);
      bp::object l = concreteData.getLabelItem(index);
      // Call numpy outer product function
      mean.attr("__iadd__")(l);
      squares.attr("__iadd__")(np.attr("outer")(l,l));
      ++sampleCount;
  }

  void GaussianAggregator::Aggregate(const GaussianAggregator& aggregator)
  {
      mean.attr("__iadd__")(aggregator.mean);
      squares.attr("__iadd__")(aggregator.squares);
      sampleCount += aggregator.sampleCount;
  }

  GaussianAggregator GaussianAggregator::DeepClone() const
  {
      GaussianAggregator result(ndims);
      result.Aggregate(*this);
      return result;
  }
}
