#include "StatisticsAggregators.h"
#include "DataPointCollection.h"

namespace Teutoburg
{
    HistogramAggregator::HistogramAggregator()
    {
        bins = bp::dict();
        sampleCount = 0;
    }

    bp::tuple HistogramAggregator::GetResponse(bp::object inp) const
    {
        bp::object keys = bins.keys();
        int max_val = 0;
        bp::object max_key;
        for(int i=0; i<bp::len(keys); ++i)
        {
            if(max_val<bp::extract<int>(bins[keys[i]]))
            {
                max_val = bp::extract<int>(bins[keys[i]]);
                max_key = keys[i];
            }
        }
        return bp::make_tuple(max_key, bins);
    }

    bp::dict HistogramAggregator::GetPyObject(void)
    {
        return bins;
    }

  int HistogramAggregator::getSampleCount() const
  {
      return sampleCount;
  }

  // IStatisticsAggregator implementation
  void HistogramAggregator::Clear()
  {
      bins = bp::dict();
      sampleCount = 0;
  }

  inline void HistogramAggregator::countUp(bp::object label, int step)
  {
      bins[label] = int(bp::extract<int>(bins.setdefault(label,0))) + step;
      sampleCount += step;
  }

  double HistogramAggregator::Entropy() const
  {
      double result = 0.0;

      bp::list vals = bins.values();
      for(int i=0; i<bp::len(vals); ++i)
      {
          double val= bp::extract<int>(vals[i]);
          if(val>0.0)
          {
              val /= double(sampleCount);
              result -= val*log2(val);
          }

      }

      return result;
  }

  void HistogramAggregator::Aggregate(const sw::IDataPointCollection& data, unsigned int index)
  {
      const DataPointCollection& concreteData = (const DataPointCollection&)(data);
      bp::object label = concreteData.getLabelItem(index);

      countUp(label);
  }

  void HistogramAggregator::Aggregate(const HistogramAggregator& aggregator)
  {
      bp::list keys = aggregator.bins.keys();
      for(int i=0; i<bp::len(keys); ++i)
      {
          countUp(keys[i], bp::extract<int>(aggregator.bins[keys[i]]));
      }
      sampleCount += aggregator.sampleCount;
  }

  HistogramAggregator HistogramAggregator::DeepClone() const
  {
      HistogramAggregator result;
      result.Aggregate(*this);
      return result;
  }


    /* GaussianAggregator */
    GaussianAggregator::GaussianAggregator(unsigned int data_dims, unsigned int label_dims)
    {
        this->data_dims = data_dims;
        this->label_dims = label_dims;

        ATy = bp::object(np.attr("zeros")(bp::make_tuple(data_dims+1, label_dims)));
        ATA = bp::object(np.attr("zeros")(bp::make_tuple(data_dims+1, data_dims+1)));
        yTy = bp::object(np.attr("zeros")(bp::make_tuple(label_dims, label_dims)));
        uptodate = false;
        sampleCount = 0;
    }

    /*
    double GaussianAggregator::Entropy() const
    {
        // Call numpy determinant function instead
        double det = bp::extract<double>(np.attr("linalg").attr("det")(cov));
        //return log(2*bp::numeric::pi*bp::numeric::e) + 0.5*log(det)
        return 2.8378770664093453 + 0.5*log(det);
    }
    */

    int GaussianAggregator::getSampleCount(void ) const
    {
        return sampleCount;
    }

    void GaussianAggregator::Clear()
    {
        ATA.attr("fill")(0);
        ATy.attr("fill")(0);
        yTy.attr("fill")(0);
        sampleCount = 0;
        uptodate = false;
    };

    void GaussianAggregator::Aggregate(const sw::IDataPointCollection& data, unsigned int index)
    {
        const DataPointCollection& concreteData = (const DataPointCollection&)(data);

        // update all variables
        uptodate = false;
        ++sampleCount;
        bp::tuple item = concreteData.getItem(index);
        bp::object xx = np.attr("hstack")(bp::make_tuple(item[0], 1));
        bp::object yy = item[1];

        ATA.attr("__iadd__")(np.attr("outer")(xx, xx));
        ATy.attr("__iadd__")(np.attr("outer")(xx, yy));
        yTy.attr("__iadd__")(np.attr("outer")(yy, yy));
    };

    void GaussianAggregator::Aggregate(const GaussianAggregator& other)
    {
        sampleCount += other.sampleCount;
        uptodate = false;
        ATA.attr("__iadd__")(other.ATA);
        ATy.attr("__iadd__")(other.ATy);
        yTy.attr("__iadd__")(other.yTy);
    };

    void GaussianAggregator::update()
    {
        if(!uptodate)
        {
            M = np.attr("linalg").attr("solve")(ATA, ATy);
            uptodate = true;
        }
    }

    bp::object GaussianAggregator::GetPyObject() const
    {
        //update();
        return np.attr("linalg").attr("solve")(ATA, ATy);
    }

    double GaussianAggregator::Entropy() const
    {
        //update();
        if(sampleCount<=1)
        {
            return 0;
        }

        bp::object M = np.attr("linalg").attr("solve")(ATA, ATy);
        bp::object MTATy = M.attr("T").attr("dot")(ATy);
        bp::object c= (((yTy.attr("__sub__")(MTATy)).attr("__sub__")(bp::object(MTATy.attr("T")))).attr("__add__")((M.attr("T").attr("dot")(ATA)).attr("dot")(M))).attr("__mul__")(1.0/(double)sampleCount);

        double E;
        if(bp::extract<int>(c.attr("ndim")) > 1)
        {
            E = 0.5*log(bp::extract<double>(np.attr("linalg").attr("det")(c.attr("__mul__")(2*bc::pi<double>()*bc::e<double>()))));
        } else
        {
            E = 0.5*log(bp::extract<double>(c.attr("__mul__")(2*bc::pi<double>()*bc::e<double>())));
        }
        return E;
    }

    GaussianAggregator GaussianAggregator::DeepClone() const
    {
        GaussianAggregator result(data_dims, label_dims);
        result.Aggregate(*this);
        return result;
    };

    bp::tuple GaussianAggregator::GetResponse(bp::object inp) const
    {
        bp::object xx = np.attr("hstack")(bp::make_tuple(inp, 1));
        bp::object M = np.attr("linalg").attr("solve")(ATA, ATy);
        bp::object MTATy = M.attr("T").attr("dot")(ATy);
        bp::object c= (((yTy.attr("__sub__")(MTATy)).attr("__sub__")(bp::object(MTATy.attr("T")))).attr("__add__")((M.attr("T").attr("dot")(ATA)).attr("dot")(M))).attr("__mul__")(1.0/(double)sampleCount);
        return bp::make_tuple(xx.attr("dot")(M), c);
    }

}
