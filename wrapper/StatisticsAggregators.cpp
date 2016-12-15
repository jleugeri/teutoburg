#include "StatisticsAggregators.h"

#include <iostream>

#include "DataPointCollection.h"
#include "boost/python/object.hpp"

namespace Teutoburg
{
    bp::object HistogramAggregator::GetPyObject(void)
    {
        return bins.attr("astype")("float").attr("__mul__")(1.0f/(double)sampleCount);
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
      int label = concreteData.getLabelItem(index);

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
}
