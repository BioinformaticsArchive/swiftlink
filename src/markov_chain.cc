using namespace std;

#include <cmath>
#include <vector>
#include <algorithm>

#include <time.h>
#include <omp.h>

#include "markov_chain.h"
#include "peel_sequence_generator.h"
#include "peeler.h"
#include "locus_sampler2.h"
#include "meiosis_sampler.h"
#include "pedigree.h"
#include "genetic_map.h"
#include "descent_graph.h"
#include "linkage_writer.h"
#include "progress.h"
#include "types.h"
#include "random.h"
#include "lod_score.h"

#ifdef USE_CUDA
  #include "gpu_lodscores.h"
#endif

#define CODA_OUTPUT 1


void MarkovChain::init() {
    for(int i = 0; i < omp_get_max_threads(); ++i) {
        LocusSampler* tmp = new LocusSampler(ped, &map, psg, i);
        lsamplers.push_back(tmp);
    }
    
    for(int i = 0; i < int(map.num_markers()); ++i) {
        l_ordering.push_back(i);
    }
    
    unsigned num_meioses = 2 * (ped->num_members() - ped->num_founders());
    
    for(unsigned int i = 0; i < num_meioses; ++i) {
        unsigned person_id = ped->num_founders() + (i / 2);
        enum parentage p = static_cast<enum parentage>(i % 2);
        
        Person* tmp = ped->get_by_index(person_id);
        
        if(not tmp->safe_to_ignore_meiosis(p)) {
            m_ordering.push_back(i);
        }
    }
    
    if(temperature != 0.0) {
        return;
    }
    
    lod = new LODscores(&map);
    
    // lod scorers
    for(int i = 0; i < omp_get_max_threads(); ++i) {
        Peeler* tmp = new Peeler(ped, &map, psg, lod);
        //tmp->set_locus(i);
        peelers.push_back(tmp);
    }
    
    double trait_prob = peelers[0]->calc_trait_prob();
    
    lod->set_trait_prob(trait_prob);
    
    printf("P(T) = %.5f\n", trait_prob / log(10));
    
#ifdef USE_CUDA
    if(options.use_gpu) {
        gpulod = new GPULodscores(ped, &map, psg, options, trait_prob);
    }
#endif
}

void MarkovChain::step_lsampler(DescentGraph& dg) {
    random_shuffle(l_ordering.begin(), l_ordering.end());
    
    vector<int> thread_assignments(omp_get_max_threads(), -1);
    vector<int> tmp(l_ordering);

    #pragma omp parallel
    {
        while(1) {
            #pragma omp critical
            {
                int locus = -1;
                if(not tmp.empty()) {
                    for(int j = int(tmp.size()-1); j >= 0; --j) {
                        if(noninterferring(thread_assignments, tmp[j])) {
                            locus = tmp[j];
                            tmp.erase(tmp.begin() + j);
                            break;
                        }
                    }
                }

                thread_assignments[omp_get_thread_num()] = locus;
            }
            
            if(thread_assignments[omp_get_thread_num()] == -1)
                break;
            
            lsamplers[omp_get_thread_num()]->set_locus_minimal(thread_assignments[omp_get_thread_num()]);
            lsamplers[omp_get_thread_num()]->step(dg, thread_assignments[omp_get_thread_num()]);
        }
    }
}

void MarkovChain::step_msampler(DescentGraph& dg) {
    random_shuffle(m_ordering.begin(), m_ordering.end());
    
    msampler.reset(dg, m_ordering[0]);
    for(unsigned int j = 0; j < m_ordering.size(); ++j) {
        msampler.step(dg, m_ordering[j]);
    }
}

void MarkovChain::score(DescentGraph& dg) {
#ifdef USE_CUDA
    if(not options.use_gpu) {
#endif
        #pragma omp parallel for
        for(int j = 0; j < int(map.num_markers() - 1); ++j) {
            peelers[omp_get_thread_num()]->set_locus(j);
            peelers[omp_get_thread_num()]->process(&dg);
        }
        
#ifdef USE_CUDA
    }
    else {
        gpulod->calculate(dg);
    }
#endif
}

void MarkovChain::step(DescentGraph& dg, int i) {
    
    if(get_random() < options.lsampler_prob) {
        step_lsampler(dg);
    }
    else {
        step_msampler(dg);
    }
    
    /*
    if(temperature != 0.0) {
        return;
    }
    */
        
#ifdef CODA_OUTPUT
    if(i < options.burnin) {
        double current_likelihood = dg.get_likelihood(&map);
        if(current_likelihood == LOG_ILLEGAL) {
            fprintf(stderr, "error: descent graph illegal...\n");
            abort();
        }
            
        fprintf(stderr, "%d\t%f\n", i+1, current_likelihood);
        //fprintf(stderr, "%f: %d\t%f\n", temperature, i+1, current_likelihood);
    }
#endif

    if(temperature != 0.0) {
        return;
    }
    
    if(i < options.burnin) {
        return;
    }
        
    if((i % options.scoring_period) == 0) {
         score(dg);
    }
}

LODscores* MarkovChain::get_lodscores() {
#ifdef USE_CUDA
    if(options.use_gpu) {
        gpulod->get_results(lod);
    }
#endif
    
    return lod;
}


LODscores* MarkovChain::run(DescentGraph& dg) {

    if(dg.get_likelihood(&map) == LOG_ILLEGAL) {
        fprintf(stderr, "error: descent graph illegal pre-markov chain...\n");
        abort();
    }

    Progress p("MCMC: ", options.iterations + options.burnin);
        
    for(int i = 0; i < (options.iterations + options.burnin); ++i) {
        
        step(dg, i);
        p.increment();
    }
    
    p.finish();
    
    return get_lodscores();
}

bool MarkovChain::noninterferring(vector<int>& x, int val) {
    for(int i = 0; i < int(x.size()); ++i) {
        if(i != omp_get_thread_num()) {
            int diff = val - x[i];
            if((diff == 1) or (diff == -1)) {
                return false;
            }
        }
    }
    return true;
}

double MarkovChain::get_likelihood(DescentGraph& dg) {
    return dg.get_likelihood(&map);
}

