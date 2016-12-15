#include "FeatureResponseFunctions.h"

#include <cmath>

#include <iostream>
#include <sstream>

#include "DataPointCollection.h"
#include "Random.h"

namespace Teutoburg
{
    bp::object LinearFeatureResponse::GetPyObject(void)
    {
        return normal;
    }

    float LinearFeatureResponse::GetResponse(bp::object data) const
    {
        return bp::extract<float>(this->normal.attr("dot")(data));
    }

    float LinearFeatureResponse::GetResponse(const sw::IDataPointCollection& data, unsigned int dataIndex) const
    {
        const Teutoburg::DataPointCollection& concreteData = (Teutoburg::DataPointCollection&)(data);
        return GetResponse(concreteData.getDataItem(dataIndex));
    }
}
