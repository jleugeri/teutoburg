#include "DataPointCollection.h"

namespace Teutoburg {
    /* Define data point collections */

    DataPointCollection::DataPointCollection(bp::list data)
    {
        _data = data;
        _Count = bp::len(_data);
        if(_Count > 0)
        {
            bp::tuple d0 = bp::tuple(_data[0]);
            _CountDims = bp::len(d0[0]);
            if(PyObject_HasAttrString(bp::object(d0[1]).ptr(), "__len__"))
            {
                _CountLabelDims = bp::len(d0[1]);
            } else {
                _CountLabelDims = 0;
            }
        } else        {
            _CountDims = 0;
            _CountLabelDims = 0;
        }
    }

    void DataPointCollection::addItem(bp::tuple datapoint)
    {
        _data.append(datapoint);
        ++_Count;
    }

    bp::object DataPointCollection::getDataItem(unsigned int ID) const
    {
        return getItem(ID)[0];
    }

    bp::object DataPointCollection::getLabelItem(unsigned int ID) const
    {
        return getItem(ID)[1];
    }

    bp::tuple DataPointCollection::getItem(unsigned int ID) const
    {
        return (bp::tuple)_data[ID];
    }

    unsigned int DataPointCollection::Count(void) const
    {
        return _Count;
    }

    unsigned int DataPointCollection::CountDims(void) const
    {
        return _CountDims;
    }

    unsigned int DataPointCollection::CountLabelDims(void) const
    {
        return _CountLabelDims;
    }
}
