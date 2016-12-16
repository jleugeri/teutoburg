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
        bp::object data;
        bp::object labels;

    public:
        bp::object getDataItem(int ID) const;
        bp::object getLabelItem(int ID) const;

        DataPointCollection(bp::object data, bp::object labels);
        unsigned int Count(void) const;
        int CountDims(void) const;
        int CountClasses(void) const;
        int CountLabelDims(void) const;
    };
}

#endif /* end of include guard: _DATAPOINTCOLLECTION_H_ */
