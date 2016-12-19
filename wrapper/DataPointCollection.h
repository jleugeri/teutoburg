#ifndef _DATAPOINTCOLLECTION_H_
#define _DATAPOINTCOLLECTION_H_

#include "include.h"
#include "DataPointCollection.h"

namespace Teutoburg {
    /* declare data point collections;
    this class uses python iterables to express data and labels/target values */
    class DataPointCollection: public sw::IDataPointCollection
    {
    private:
        unsigned int _Count;
        unsigned int _CountDims;
        unsigned int _CountLabelDims;
        bp::list _data;

    public:
        bp::tuple getItem(unsigned int ID) const;
        bp::object getDataItem(unsigned int ID) const;
        bp::object getLabelItem(unsigned int ID) const;

        void addItem(bp::tuple datapoint);

        DataPointCollection(bp::list data);
        unsigned int Count(void) const;
        unsigned int CountDims(void) const;
        unsigned int CountLabelDims(void) const;
    };
}

#endif /* end of include guard: _DATAPOINTCOLLECTION_H_ */
