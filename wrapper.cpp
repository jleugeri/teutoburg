#include <boost/python.hpp>
//#include <boost/numpy.hpp>
#include <stdio.h>
#include "Sherwood.h"
#include "DataPointCollection.h"
#include "StatisticsAggregators.h"
#include "FeatureResponseFunctions.h"
#include "TrainingContexts.h"
#include "TrainingContexts.cpp"
#include "Teutoburg.h"
#include <numpy/ndarrayobject.h>

using namespace boost::python;

namespace sw = MicrosoftResearch::Cambridge::Sherwood;
namespace bp = boost::python;
namespace bu = boost::numeric::ublas;


int init_numpy()
{
    import_array();
}

namespace Teutoburg
{
    // For some reason, these definitions must be in the same translation unit as the "import_array()" directive...
    LinearFeatureResponse::LinearFeatureResponse(npy_intp dims)
    {
        npy_intp tmp = dims;
        PyObject* obj = PyArray_ZEROS(1, &tmp, NPY_DOUBLE, 0);
        bp::handle<> arr(obj);
        normal = bp::object(arr);
    }

    LinearFeatureResponse::LinearFeatureResponse(sw::Random& random, npy_intp dims)
    {
        npy_intp tmp = dims;
        PyObject* obj = PyArray_ZEROS(1, &tmp, NPY_DOUBLE, 0);

        for(npy_intp i=0; i<dims; ++i)
        {
            *((double*)PyArray_GETPTR1(obj, i)) = random.NextDouble()*2-1.0;
        }

        bp::handle<> arr(obj);
        normal = bp::object(arr);
    }

    HistogramAggregator::HistogramAggregator(int nClasses)
    {
        this->nClasses = nClasses;

        npy_intp tmp = nClasses;

        if(nClasses > 0)
        {
            PyObject* obj = PyArray_ZEROS(1, &tmp, NPY_INT, 0);
            bp::handle<> arr(obj);
            bins = bp::object(arr);
        }

        sampleCount = 0;
    }
}

BOOST_PYTHON_MODULE(sherwood)
{
    init_numpy();
    /*
    class_<sw::Node<sw::LinearFeatureResponse2d,sw::HistogramAggregator>>("Node")
        .def_readonly("feature", &sw::Node<sw::LinearFeatureResponse2d,sw::HistogramAggregator>::Feature)
        .def_readonly("threshold", &sw::Node<sw::LinearFeatureResponse2d,sw::HistogramAggregator>::Threshold)
        .add_property("split", &sw::Node<sw::LinearFeatureResponse2d,sw::HistogramAggregator>::IsSplit)
        .add_property("leaf", &sw::Node<sw::LinearFeatureResponse2d,sw::HistogramAggregator>::IsLeaf)
    ;
    */

    using F = Teutoburg::LinearFeatureResponse;
    using S = Teutoburg::HistogramAggregator;
    def("trainForest", Teutoburg::train<F,S>, return_value_policy<manage_new_object>());

    class_<Teutoburg::Node<F,S>, boost::noncopyable>("Node", no_init)
        .add_property("isSplit", &Teutoburg::Node<F,S>::IsSplit)
        .add_property("isLeaf", &Teutoburg::Node<F,S>::IsLeaf)
        .add_property("threshold", &Teutoburg::Node<F,S>::GetThreshold)
        .add_property("feature", &Teutoburg::Node<F,S>::GetFeature)
        .add_property("statistics", &Teutoburg::Node<F,S>::GetStatistics)
    ;

    class_<Teutoburg::Tree<F,S>, boost::noncopyable>("Tree", no_init)
        .def("__len__", &Teutoburg::Tree<F,S>::NodeCount)
        .def("__getitem__", &Teutoburg::Tree<F,S>::GetNode, return_value_policy<manage_new_object>())
    ;

    class_<Teutoburg::Forest<F,S>, boost::noncopyable>("Forest", no_init)
        .def("__len__", &Teutoburg::Forest<F,S>::TreeCount)
        .def("__getitem__", &Teutoburg::Forest<F,S>::GetTree, return_value_policy<manage_new_object>())
        .def("__call__", &Teutoburg::Forest<F,S>::Apply) 
    ;


/*
    sw::Random rnd;

    bu::vector<bu::vector<float>> dd(10);
    bu::vector<unsigned int> ll(10);
    for(int i=0; i<10; ++i)
    {
        dd[i].resize(5);
        for(int j=0; j<5; ++j)
            dd[i][j] = i+j;
        ll[i] = i % 3;
    }

    Teutoburg::DataPointCollection d = Teutoburg::DataPointCollection(dd, ll);
    sw::TrainingParameters p = sw::TrainingParameters();

    Teutoburg::ClassificationTrainingContext<Teutoburg::LinearFeatureResponse> c = Teutoburg::ClassificationTrainingContext<Teutoburg::LinearFeatureResponse>(5,d.CountClasses());
    Teutoburg::LinearFeatureResponse r = c.GetRandomFeature(rnd);
    Teutoburg::HistogramAggregator h = c.GetStatisticsAggregator();

    for(int i=0; i<10; ++i)
    {
        h.Aggregate(d, i);
    }

    std::cout << "Response: " << r.GetResponse(d, 8) << "; Entropy: " << h.Entropy();


    std::auto_ptr<sw::Forest<Teutoburg::LinearFeatureResponse, Teutoburg::HistogramAggregator>> forest = sw::ForestTrainer<Teutoburg::LinearFeatureResponse, Teutoburg::HistogramAggregator>::TrainForest(rnd, p, c, d );

    std::vector<std::vector<int>> leafNodeIndices;
    forest->Apply(d, leafNodeIndices);

    for(int i=0; i<1; ++i)
    for(int j=0; j<10; ++j)
    {
        std::cout << leafNodeIndices[i][j] << " ";
    }
    for(int i=0; i<5; ++i)
    {
        std::cout << (Teutoburg::Node<Teutoburg::LinearFeatureResponse,Teutoburg::HistogramAggregator>)forest->GetTree(0).GetNode(i) << ".";
    }

    Teutoburg::Tree<Teutoburg::LinearFeatureResponse,Teutoburg::HistogramAggregator> T = Teutoburg::Tree<Teutoburg::LinearFeatureResponse,Teutoburg::HistogramAggregator>(forest->GetTree(0));
    std::cout << Teutoburg::Tree<Teutoburg::LinearFeatureResponse,Teutoburg::HistogramAggregator>(forest->GetTree(0)).toStr();

    class_<Teutoburg::DataPointCollection>("data", bp::init<bp::object>())
        .def("__len__", &Teutoburg::DataPointCollection::Count)
        .def("__getitem__", &Teutoburg::DataPointCollection::getItem)
        .def_readonly("data", &Teutoburg::DataPointCollection::getData)
        .def_readonly("labels", &Teutoburg::DataPointCollection::getLabels)
    ;


    // Get handles to two overloaded implementations of GetResponse
    float (Teutoburg::LinearFeatureResponse::*ftmp_1)(const bp::object, unsigned int) const =  &Teutoburg::LinearFeatureResponse::GetResponse;
    float (Teutoburg::LinearFeatureResponse::*ftmp_2)(const bp::object) const = &Teutoburg::LinearFeatureResponse::GetResponse;
    class_<Teutoburg::LinearFeatureResponse>("LinearFeatureResponse", bp::init<bp::object>())
        .def("__call__", ftmp_1)
        .def("__call__", ftmp_2)
    ;*/
}
