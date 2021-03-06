#ifndef _TEUTOBURG_H_
#define _TEUTOBURG_H_

#include "Node.h"
#include "DataPointCollection.h"
#include "FeatureResponseFunctions.h"
#include "StatisticsAggregators.h"


namespace sw = MicrosoftResearch::Cambridge::Sherwood;
namespace bp = boost::python;

namespace Teutoburg
{
    bp::object np;

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

        bp::object GetResponse(bp::object inp) const
        {
            return node_->TrainingDataStatistics.GetResponse(inp);
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

        bp::list Apply(bp::list data)
        {
            std::vector<std::vector<int>> leafNodeIndices;
            bp::list all_stats;
            DataPointCollection d = DataPointCollection(data);
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
                        bp::list tree_stats;
                        all_stats.append(tree_stats);
                    }
                    all_stats[k].attr("append")(forest_->GetTree(t).GetNode(*it2).TrainingDataStatistics.GetResponse(d.getDataItem(k)));
                }
            }

            return all_stats;
        }
    };

    // factory that creates forsts objects through training (ger.: "Baumschule" ;) )
    template<class F, class S>// where F : IFeatureResponse where S: IStatisticsAggregator<S>
    Forest<F,S>* trainClassification(bp::list data, int numTrees, int numFeatures, int numThresholds, int maxLevels, bool verbose)
    {
        sw::Random rnd;

        sw::TrainingParameters p = sw::TrainingParameters();
        p.NumberOfTrees = numTrees;
        p.NumberOfCandidateFeatures = numFeatures;
        p.NumberOfCandidateThresholdsPerFeature = numThresholds;
        p.MaxDecisionLevels = maxLevels;
        p.Verbose = verbose;

        Teutoburg::DataPointCollection d = Teutoburg::DataPointCollection(data);
        Teutoburg::ClassificationTrainingContext<F> c = ClassificationTrainingContext<F>(d.CountDims());
        std::auto_ptr<sw::Forest<F,S>> forest = sw::ForestTrainer<F,S>::TrainForest(rnd, p, c, d );
        return new Forest<F,S>(forest);
    }

    // factory that creates forsts objects through training (ger.: "Baumschule" ;) )
    template<class F, class S>// where F : IFeatureResponse where S: IStatisticsAggregator<S>
    Forest<F,S>* trainRegression(bp::list data, int numTrees, int numFeatures, int numThresholds, int maxLevels, bool verbose)
    {
        sw::Random rnd;

        sw::TrainingParameters p = sw::TrainingParameters();
        p.NumberOfTrees = numTrees;
        p.NumberOfCandidateFeatures = numFeatures;
        p.NumberOfCandidateThresholdsPerFeature = numThresholds;
        p.MaxDecisionLevels = maxLevels;
        p.Verbose = verbose;

        Teutoburg::DataPointCollection d = Teutoburg::DataPointCollection(data);
        Teutoburg::RegressionTrainingContext<F> c = RegressionTrainingContext<F>(d.CountDims(), d.CountLabelDims());
        std::auto_ptr<sw::Forest<F,S>> forest = sw::ForestTrainer<F,S>::TrainForest(rnd, p, c, d );

        return new Forest<F,S>(forest);
    }
}

#endif /* end of include guard: _TEUTOBURG_H_ */
