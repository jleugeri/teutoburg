#include "DataPointCollection.h"
#include <boost/python/tuple.hpp>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>

namespace Teutoburg {
    /* Define data point collections */

    DataPointCollection::DataPointCollection(bp::object data, bp::object labels)
    {

        this->data = data;
        this->labels = labels;
    }

    bp::object DataPointCollection::getDataItem(int ID) const
    {
        return this->data.attr("__getitem__")(ID);
    }

    int DataPointCollection::getLabelItem(int ID) const
    {
        return bp::extract<int>(this->labels.attr("__getitem__")(ID).attr("item")());
    }

    unsigned int DataPointCollection::Count(void) const
    {
        return bp::extract<unsigned int>((this->data.attr("__len__")()));
    }

    int DataPointCollection::CountDims(void) const
    {
        return (Count() <= 0) ? (int)0 : (int)bp::extract<int>(this->data.attr("__getitem__")(0).attr("__len__")());
    }

    int DataPointCollection::CountClasses(void) const
    {
        return bp::extract<int>((this->labels.attr("max")().attr("__add__")(1).attr("item")()));
    }
}
