#include "Node.h"
#include "Sherwood.h"
#include "FeatureResponseFunctions.h"
#include "StatisticsAggregators.h"
#include <boost/shared_ptr.hpp>


namespace sw = MicrosoftResearch::Cambridge::Sherwood;
namespace bp = boost::python;

namespace Teutoburg
{
    /* Forward declaration of required classes*/
    template<class F, class S> class Node;
    template<class F, class S> class Tree;
    template<class F, class S> class Forest;

    inline unsigned int mylog2(unsigned int i)
    {
        unsigned int ret = 0;
        while (i>>=1) ++ret;
        return ret;
    }

    template<class F, class S>// where F : IFeatureResponse where S: IStatisticsAggregator<S>
    class Node
    {
        // Classical pointer; ownership of the node remains with the tree it belongs to
        sw::Node<F,S> *node_;
    public:
        Node(sw::Node<F,S> *node)
        {
            node_ = node;
        }

        bool IsSplit(void) const
        {
            return node_->IsSplit();
        }

        bool IsLeaf(void) const
        {
            return node_->IsLeaf();
        }

        float GetThreshold(void) const
        {
            return node_->Threshold;
        }

        bp::object GetFeature(void) const
        {
            return node_->Feature.GetPyObject();
        }

        bp::object GetStatistics(void) const
        {
            return node_->TrainingDataStatistics.GetPyObject();
        }
    };

    template<class F, class S>
    class Tree // where F:IFeatureResponse where S:IStatisticsAggregator<S>
    {
        // Classical pointer; ownership of the tree remains with the forest it belongs to
        sw::Tree<F,S> *tree_;
    public:

        Tree(sw::Tree<F,S>* tree)
        {
            tree_ = tree;
        }

        unsigned int NodeCount()
        {
            return (tree_ == NULL) ? 0 : tree_->NodeCount();
        }

        Node<F,S>* GetNode(int ID)
        {
            // TODO: assert index OK
            return new Node<F,S>(&tree_->GetNode(ID));
        }
    };

    template<class F, class S>// where F : IFeatureResponse where S: IStatisticsAggregator<S>
    class Forest // where F:IFeatureResponse and S:IStatisticsAggregator<S>
    {
        // Managed pointer that takes ownership of the object
        std::auto_ptr<sw::Forest<F,S>> forest_;
    public:
        Forest(std::auto_ptr<sw::Forest<F,S>> forest)
        {
            forest_ = forest;
        }

        unsigned int TreeCount()
        {
            return forest_.get() == NULL ? 0 : forest_->TreeCount();
        }

        Tree<F,S>* GetTree(int ID)
        {
            return new Tree<F,S>(&forest_->GetTree(ID));
        }

        /*
        bp::object FindLeaves(bp::object data)
        {
            std::vector<std::vector<int>> leafNodeIndices;
            DataPointCollection d = DataPointCollection(data, bp::object());
            forest_->Apply(d, leafNodeIndices);


        }*/

        bp::list Apply(bp::object data)
        {
            std::vector<std::vector<int>> leafNodeIndices;
            std::vector<S> all_stats;
            DataPointCollection d = DataPointCollection(data, bp::object());
            forest_->Apply(d, leafNodeIndices);

            int t=0;
            for(std::vector<std::vector<int>>::const_iterator it1 = leafNodeIndices.begin();
                it1 != leafNodeIndices.end();
                ++it1, ++t)
            {
                int k=0;
                for(std::vector<int>::const_iterator it2 = it1->begin();
                    it2 != it1->end();
                    ++it2, ++k)
                {
                    if(t==0)
                    {
                        S stats = forest_->GetTree(t).GetNode(*it2).TrainingDataStatistics.DeepClone();
                        all_stats.push_back(stats);
                    }
                    else
                    {
                        all_stats[k].Aggregate(forest_->GetTree(t).GetNode(*it2).TrainingDataStatistics);
                    }
                }
            }

            bp::list all_stats_py;
            for(typename std::vector<S>::iterator it = all_stats.begin(); it!=all_stats.end(); ++it)
            {
                all_stats_py.append(it->GetPyObject());
            }

            return all_stats_py;
        }
    };

    // factory that creates forsts objects through training (ger.: "Baumschule" ;) )
    template<class F, class S>// where F : IFeatureResponse where S: IStatisticsAggregator<S>
    Forest<F,S>* train(bp::object data, bp::object labels, int numTrees, int numFeatures, int numThresholds, int maxLevels, bool verbose)
    {
        sw::Random rnd;

        sw::TrainingParameters p = sw::TrainingParameters();
        p.NumberOfTrees = numTrees;
        p.NumberOfCandidateFeatures = numFeatures;
        p.NumberOfCandidateThresholdsPerFeature = numThresholds;
        p.MaxDecisionLevels = maxLevels;
        p.Verbose = verbose;

        Teutoburg::DataPointCollection d = Teutoburg::DataPointCollection(data, labels);
        Teutoburg::ClassificationTrainingContext<F> c = ClassificationTrainingContext<F>(d.CountDims(), d.CountClasses());
        std::auto_ptr<sw::Forest<F,S>> forest = sw::ForestTrainer<F,S>::TrainForest(rnd, p, c, d );
        return new Forest<F,S>(forest);
    }
}
