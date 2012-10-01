using namespace std;

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <iostream>
#include <fstream>

#include "descent_graph.h"
#include "linkage_program.h"
#include "linkage_writer.h"
#include "disease_model.h"
#include "markov_chain.h"
#include "genetic_map.h"
#include "pedigree.h"
#include "peeler.h"
#include "random.h"
#include "sequential_imputation.h"
#include "progress.h"
//#include "gpu_markov_chain.h"
#include "lod_score.h"


bool LinkageProgram::run() {
    vector<LODscores*> all_scores;
    LODscores* tmp;
    bool ret = true;
    LinkageWriter lw(&map, outfile, options.verbose);
    
    /*
    if(options.verbose) {
        fprintf(stderr, "%s\n", dm.debug_string().c_str());
        fprintf(stderr, "%s\n", map.debug_string().c_str());
    }
    */
        
    init_random();
    if(options.random_filename == "") {
        seed_random_implicit();
    }
    else {
        seed_random_explicit(options.random_filename);
    }
    
    for(unsigned int i = 0; i < pedigrees.size(); ++i) {
        
        if(options.verbose) {
            fprintf(stderr, "%s\n", pedigrees[i].debug_string().c_str());
        }
        
        
        // it cannot actually be NULL, the program will call
        // abort() at the slightest hint of a problem
        if((tmp = run_pedigree(pedigrees[i])) == NULL) {
            fprintf(stderr, "error: pedigree '%s' failed\n", pedigrees[i].get_id().c_str());
            
            ret = false;
            goto die;
        }
        
        all_scores.push_back(tmp);
    }
    
    //LinkageWriter lw(&map, outfile, options.verbose);
    if(not lw.write(all_scores)) {
        fprintf(stderr, "error: could not write output file '%s'\n", outfile.c_str());
        
        ret = false;
        goto die;
    }
    
die:
    for(unsigned int i = 0; i < all_scores.size(); ++i) {
        delete all_scores[i];
    }
    
    return ret;
}

LODscores* LinkageProgram::run_pedigree(Pedigree& p) {
    
    if(options.verbose) {
        fprintf(stderr, "processing pedigree %s\n", p.get_id().c_str());
    }
    
    
    DescentGraph dg(&p, &map);
    
    dg.random_descentgraph(); // just in case the user selects zero sequential imputation iterations
    
    if(dg.get_likelihood(&map) == LOG_ZERO) {
        fprintf(stderr, "error, bad descent graph %s:%d\n", __FILE__, __LINE__);
        abort();
    }
    
    
    PeelSequenceGenerator psg(&p, &map, options.verbose);
    psg.build_peel_sequence(options.peelopt_iterations);
    
    /*
    SequentialImputation si(&p, &map, &psg);
    //si.run(dg, options.si_iterations);
    si.parallel_run(dg, options.si_iterations);
    
    if(not options.use_gpu) {
        MarkovChain chain(&p, &map, &psg, options);
        return chain.run(dg);
    }
    else {
        GPUMarkovChain chain(&p, &map, &psg, options);
        return chain.run(dg);
    }
    
    fprintf(stderr, "error: nothing was run (%s:%d)\n", __FILE__, __LINE__);
    abort();
    */
    
    SequentialImputation si(&p, &map, &psg);
    si.parallel_run(dg, options.si_iterations);
    
    MarkovChain chain(&p, &map, &psg, options);
    return chain.run(dg);
    
    /*
    int num_chains = 5;
    double temperatures[] = { 0.0, 0.1, 0.2, 0.3, 0.4 };
    vector<DescentGraph*> graphs;
    vector<MarkovChain*> chains;
    SequentialImputation si(&p, &map, &psg);
    
    for(int i = 0; i < num_chains; ++i) {
        DescentGraph* dg = new DescentGraph(&p, &map);
        
        si.parallel_run(*dg, options.si_iterations);
        
        graphs.push_back(dg);
        
        MarkovChain* mc = new MarkovChain(&p, &map, &psg, options, temperatures[i]);
        chains.push_back(mc);
    }
    
    
    Progress prog("Tempering: ", options.iterations + options.burnin);
        
    for(int i = 0; i < (options.iterations + options.burnin); ++i) {
        for(int j = 0; j < num_chains; ++j) {
            chains[j]->step(*graphs[j], i);
        }
        
        int c = get_random_int(num_chains-1);
        
        double pii = chains[ c ]->get_likelihood(*graphs[ c ]);
        double pjj = chains[c+1]->get_likelihood(*graphs[c+1]);
        double pij = chains[ c ]->get_likelihood(*graphs[c+1]);
        double pji = chains[c+1]->get_likelihood(*graphs[ c ]);
        
        double metrop = (pij + pji) - (pii + pjj);
        
        //fprintf(stderr, "%d <--> %d : ", c, c+1);
        
        if(get_random() < exp(min(0.0, metrop))) {
            DescentGraph* tmp = graphs[c];
            graphs[c] = graphs[c+1];
            graphs[c+1] = tmp;
            
            //fprintf(stderr, "swapped\n");
        }
        else {
            //fprintf(stderr, "not swapped (metrop = %.4f, pii = %.4f, pjj = %.4f, pij = %.4f, pji = %.4f)\n", metrop, pii, pjj, pij, pji);
        }
        
        prog.increment();
    }
    
    prog.finish();
    
    LODscores* lod = chains[0]->get_lodscores();
    
    for(int i = 0; i < num_chains; ++i) {
        delete graphs[i];
        delete chains[i];
    }
    
    return lod;
    */
}
