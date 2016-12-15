#pragma once

#include <string>

#include "Sherwood.h"

#include "Python.h"
#include <numpy/arrayobject.h>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/numpy.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <numpy/ndarrayobject.h>

namespace bp = boost::python;
namespace bu = boost::numeric::ublas;
namespace sw = MicrosoftResearch::Cambridge::Sherwood;

namespace Teutoburg
{
    class LinearFeatureResponse: public sw::IFeatureResponse
    {
        bp::object normal;
    public:
        LinearFeatureResponse(npy_intp dims = 0);
        LinearFeatureResponse(sw::Random& random, npy_intp dims);
        bp::object GetPyObject(void);
        float GetResponse(const bp::object data) const;
        float GetResponse(const sw::IDataPointCollection& data, unsigned int dataIndex) const;
    };
}
