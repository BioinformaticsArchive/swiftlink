using namespace std;

#include <vector>
#include <cmath>

#include "types.h"
#include "peel_sequence_generator.h"
#include "descent_graph.h"
#include "pedigree.h"
#include "genetic_map.h"
#include "sampler_rfunction.h"
#include "locus_sampler2.h"


void LocusSampler::init_rfunctions(PeelSequenceGenerator* psg) {
    vector<PeelOperation>& ops = psg->get_peel_order();
    
    rfunctions.reserve(ops.size()); // need to do this otherwise pointers may not work later...
    
    for(unsigned int i = 0; i < ops.size(); ++i) {
        vector<unsigned int>& prev_indices = ops[i].get_prevfunctions();
        vector<Rfunction*> prev_pointers;
        
        for(unsigned int j = 0; j < prev_indices.size(); ++j) {
            prev_pointers.push_back(&(rfunctions[prev_indices[j]]));
        }
        
        rfunctions.push_back(SamplerRfunction(ped, map, locus, &(ops[i]), prev_pointers));
    }
}

unsigned LocusSampler::sample_hetero_mi(enum trait allele, enum phased_trait trait) {
    if(allele == TRAIT_U) {
        return (trait == TRAIT_UA) ? 0 : 1;
    }
    else {
        return (trait == TRAIT_UA) ? 1 : 0;
    }
}

// find prob of setting mi to 0
// find prob of setting mi to 1
// normalise + sample
unsigned LocusSampler::sample_homo_mi(DescentGraph& dg, unsigned personid, enum parentage parent) {
    double prob_dist[2];
    
    prob_dist[0] = 1.0;
    prob_dist[1] = 1.0;
    
    if((locus != 0) and (not ignore_left)) {
        prob_dist[0] *= ((dg.get(personid, locus - 1, parent) == 0) ? map->get_inversetheta(locus - 1) : map->get_theta(locus - 1));
        prob_dist[1] *= ((dg.get(personid, locus - 1, parent) == 1) ? map->get_inversetheta(locus - 1) : map->get_theta(locus - 1));
    }
    
    if((locus != (map->num_markers() - 1)) and (not ignore_right)) {
        prob_dist[0] *= ((dg.get(personid, locus + 1, parent) == 0) ? map->get_inversetheta(locus) : map->get_theta(locus));
        prob_dist[1] *= ((dg.get(personid, locus + 1, parent) == 1) ? map->get_inversetheta(locus) : map->get_theta(locus));
    }
    
    return (get_random() < (prob_dist[0] / (prob_dist[0] + prob_dist[1]))) ? 0 : 1;
}

// if a parent is heterozygous, then there is one choice of meiosis indicator
// if a parent is homozygous, then sample based on meiosis indicators to immediate left and right    
unsigned LocusSampler::sample_mi(DescentGraph& dg, \
                                 enum trait allele, \
                                 enum phased_trait trait, \
                                 unsigned personid, \
                                 enum parentage parent) {
    switch(trait) {
        case TRAIT_UA:
        case TRAIT_AU:
            return sample_hetero_mi(allele, trait);
            
        case TRAIT_UU:
        case TRAIT_AA:
            return sample_homo_mi(dg, personid, parent);
            
        default:
            break;
    }
    
    abort();
}

// sample meiosis indicators
// if a parent is heterozygous, then there is one choice of meiosis indicator
// if a parent is homozygous, then sample based on meiosis indicators to immediate left and right
void LocusSampler::sample_meiosis_indicators(vector<int>& pmk, DescentGraph& dg) {
    for(unsigned i = 0; i < ped->num_members(); ++i) {
        Person* p = ped->get_by_index(i);
        
        if(p->isfounder()) {
            continue;
        }
        
        enum phased_trait mat_trait = static_cast<enum phased_trait>(pmk[p->get_maternalid()]);
        enum phased_trait pat_trait = static_cast<enum phased_trait>(pmk[p->get_paternalid()]);
        
        enum phased_trait trait = static_cast<enum phased_trait>(pmk[i]);
        enum trait mat_allele = ((trait == TRAIT_UU) or (trait == TRAIT_UA)) ? TRAIT_U : TRAIT_A;
        enum trait pat_allele = ((trait == TRAIT_UU) or (trait == TRAIT_AU)) ? TRAIT_U : TRAIT_A;
        
        unsigned mat_mi = sample_mi(dg, mat_allele, mat_trait, i, MATERNAL);
        unsigned pat_mi = sample_mi(dg, pat_allele, pat_trait, i, PATERNAL);
        
        dg.set(i, locus, MATERNAL, mat_mi);
        dg.set(i, locus, PATERNAL, pat_mi);        
    }
}

void LocusSampler::step(DescentGraph& dg, unsigned parameter) {
    
    //set_locus_minimal(parameter); // XXX doing this here messes up the sequential imputation bits
    
    // forward peel
    for(unsigned int i = 0; i < rfunctions.size(); ++i) {
        rfunctions[i].evaluate(&dg, 0);
    }
    
    vector<int> pmk(ped->num_members(), -1);
    
    // reverse peel, sampling ordered genotypes
    for(int i = static_cast<int>(rfunctions.size()) - 1; i >= 0; --i) {
        rfunctions[i].sample(pmk);
    }
    
    sample_meiosis_indicators(pmk, dg);
}

void LocusSampler::reset() {
    set_locus(locus, false, false);
}

void LocusSampler::set_locus(unsigned int l, bool left, bool right) {
    locus = l;
    ignore_left = left;
    ignore_right = right;

    for(unsigned int i = 0; i < rfunctions.size(); ++i) {
        rfunctions[i].set_locus(locus, ignore_left, ignore_right);
    }
}

void LocusSampler::set_locus_minimal(unsigned int l) {
    locus = l;
    
    for(unsigned int i = 0; i < rfunctions.size(); ++i) {
        rfunctions[i].set_locus_minimal(locus);
    }
}

double LocusSampler::locus_by_locus(DescentGraph& dg) {
    unsigned int starting_locus = 0;
    unsigned int tmp = locus;
    double weight = 0.0;
    
    
    set_locus(starting_locus, true, true);
    step(dg, starting_locus);
    weight += log(rfunctions.back().get_result());
    
    // iterate right through the markers
    for(int i = (starting_locus + 1); i < int(map->num_markers()); ++i) {
        set_locus(i, true, true);
        step(dg, i);
        weight += log(rfunctions.back().get_result());
    }
    
    // reset, in case not used for more si
    set_locus(tmp, false, false);
    
    return weight;    
}

double LocusSampler::sequential_imputation(DescentGraph& dg) {
    unsigned int starting_locus = get_random_locus();
    unsigned int tmp = locus;
    double weight = 0.0;
    
    set_locus(starting_locus, true, true);
    step(dg, starting_locus);
    weight += log(rfunctions.back().get_result());
    
    // iterate left through the markers
    for(int i = (starting_locus - 1); i >= 0; --i) {
        set_locus(i, true, false);
        step(dg, i);
        weight += log(rfunctions.back().get_result());
    }
    
    // iterate right through the markers
    for(int i = (starting_locus + 1); i < int(map->num_markers()); ++i) {
        set_locus(i, false, true);
        step(dg, i);
        weight += log(rfunctions.back().get_result());
    }
    
    // reset, in case not used for more si
    set_locus(tmp, false, false);
    
    return weight;    
}

