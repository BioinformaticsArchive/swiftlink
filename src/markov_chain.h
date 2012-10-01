#ifndef LKG_MARKOVCHAIN_H_
#define LKG_MARKOVCHAIN_H_

using namespace std;

#include "types.h"
#include "genetic_map.h"
#include "peeler.h"
#include "peel_sequence_generator.h"
#include "locus_sampler2.h"
#include "meiosis_sampler.h"
#include "pedigree.h"
#include "descent_graph.h"
#include "lod_score.h"


class MarkovChain {
    
    Pedigree* ped;
    GeneticMap map;
    PeelSequenceGenerator* psg;
    struct mcmc_options options;
    double temperature;
    vector<int> l_ordering;
    vector<int> m_ordering;
    vector<Peeler*> peelers;
#ifdef USE_CUDA
    GPULodscores* gpulod;
#endif
    LODscores* lod;
    vector<LocusSampler*> lsamplers;
    MeiosisSampler msampler;

    bool noninterferring(vector<int>& x, int val);
    
 public :
    MarkovChain(Pedigree* ped, GeneticMap* m, PeelSequenceGenerator* psg, struct mcmc_options options, double t = 0.0) :
        ped(ped), 
        map(*m, t),
        psg(psg),
        options(options),
        temperature(t),
        l_ordering(),
        m_ordering(),
        peelers(),
#ifdef USE_CUDA
        gpulod(0),
#endif
        lod(0),
        lsamplers(),
        msampler(ped, &map) {
        
        init();
    }
    
    ~MarkovChain() {
        for(unsigned int i = 0; i < lsamplers.size(); ++i) {
            delete lsamplers[i];
        }
        
        for(unsigned int i = 0; i < peelers.size(); ++i) {
            delete peelers[i];
        }
        
#ifdef USE_CUDA
        if(options.use_gpu) {
            delete gpulod;
        }
#endif
    }
    
    
    MarkovChain(const MarkovChain& rhs) :
        ped(rhs.ped), 
        map(rhs.map),
        psg(rhs.psg), 
        options(rhs.options),
        temperature(rhs.temperature),
        l_ordering(rhs.l_ordering),
        m_ordering(rhs.m_ordering),
        peelers(rhs.peelers),
#ifdef USE_CODA
        gpulod(rhs.gpulod), // XXX !
#endif
        lod(rhs.lod), // XXX !
        lsamplers(rhs.lsamplers),
        msampler(rhs.msampler) {}
    
    MarkovChain& operator=(const MarkovChain& rhs) {
        if(this != &rhs) {
            ped = rhs.ped;
            map = rhs.map;
            psg = rhs.psg;
            options = rhs.options;
            temperature = rhs.temperature;
            l_ordering = rhs.l_ordering;
            m_ordering = rhs.m_ordering;
            peelers = rhs.peelers;
#ifdef USE_CUDA
            gpulod = rhs.gpulod; // XXX !
#endif
            lod = rhs.lod; // XXX !
            lsamplers = rhs.lsamplers;
            msampler = rhs.msampler;
        }
        return *this;
    }
    
    
    double get_likelihood(DescentGraph& dg);
    
    LODscores* run(DescentGraph& dg);
    
    void init();
    void step(DescentGraph& dg, int i);
    void step_lsampler(DescentGraph& dg);
    void step_msampler(DescentGraph& dg);
    void score(DescentGraph& dg);
    LODscores* get_lodscores();
    
};

#endif

