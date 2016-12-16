#ifndef _INCLUDE_H_
#define _INCLUDE_H_

#include "Sherwood.h"
#include <boost/python.hpp>
#include "boost/python/object.hpp"
#include <numpy/ndarrayobject.h>


namespace bp = boost::python;
namespace sw = MicrosoftResearch::Cambridge::Sherwood;

namespace Teutoburg
{
    extern bp::object np;
}

#endif /* end of include guard: _INCLUDE_H_ */
