#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "Sherwood.h"

#include <boost/python.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <numpy/ndarrayobject.h>

namespace bp = boost::python;
namespace bu = boost::numeric::ublas;
namespace sw = MicrosoftResearch::Cambridge::Sherwood;

namespace Teutoburg {
    /* declare data point collections;
    this class uses python iterables to express data and labels/target values */
    class DataPointCollection: public sw::IDataPointCollection
    {
    private:
        bp::object data;
        bp::object labels;

    public:
        bp::object getDataItem(int ID) const;
        int getLabelItem(int ID) const;

        DataPointCollection(bp::object data, bp::object labels);
        unsigned int Count(void) const;
        int CountClasses(void) const;
        int CountDims(void) const;
    };
}
