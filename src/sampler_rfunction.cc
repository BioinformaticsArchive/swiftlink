using namespace std;

#include <cstdlib>

#include "sampler_rfunction.h"
#include "descent_graph.h"
#include "genetic_map.h"
#include "peeling.h"
#include "peel_matrix.h"
#include "random.h"
    

double SamplerRfunction::get_trait_probability(unsigned person_id, enum phased_trait pt) {
    
    Person* p = ped->get_by_index(person_id);
    
    // penetrace prob
    if(not p->isfounder()) {
        if(p->istyped()) {
            switch(p->get_marker(locus)) {
                case HETERO :
                    return ((pt == TRAIT_AU) or (pt == TRAIT_UA)) ? 1.0 : 0.0;
                case HOMOZ_A :
                    return (pt == TRAIT_UU) ? 1.0 : 0.0;
                case HOMOZ_B :
                    return (pt == TRAIT_AA) ? 1.0 : 0.0;
                default :
                    return 1.0;
            }
        }
        else {
            return 1.0;
        }
    }
    
    // penetrance + founder prob
    if(p->istyped()) {
        switch(p->get_marker(locus)) {
            case HETERO :
                return ((pt == TRAIT_AU) or (pt == TRAIT_UA)) ? map->get_prob(locus, pt) : 0.0;
            case HOMOZ_A :
                return (pt == TRAIT_UU) ? map->get_prob(locus, pt) : 0.0;
            case HOMOZ_B :
                return (pt == TRAIT_AA) ? map->get_prob(locus, pt) : 0.0;
            default :
                return map->get_prob(locus, pt);
        }
    }
    else {
        return map->get_prob(locus, pt);
    }
    
    fprintf(stderr, "error: %s:%d\n", __FILE__, __LINE__);
    abort();
}

double SamplerRfunction::get_recombination_probability(DescentGraph* dg, 
                                     unsigned person_id, 
                                     enum phased_trait parent_trait, 
                                     enum phased_trait kid_trait, 
                                     enum parentage parent) {
    
    enum trait t = get_trait(kid_trait, parent);
    
    // just transmission prob
    if(ignore_left and ignore_right) {
        switch(parent_trait) {
            case TRAIT_AA:
                return (t == TRAIT_A) ? 1.0 : 0.0;
            case TRAIT_UU:
                return (t == TRAIT_U) ? 1.0 : 0.0;
            case TRAIT_AU:
            case TRAIT_UA:
                return 0.5;
        }
    }
    
    // recombination + transmission prob
    
    //enum trait t = get_trait(kid_trait, parent);
    double tmp = 1.0;
    
    // deal with homozygotes first
    if(parent_trait == TRAIT_AA) {
        //return (t == TRAIT_A) ? 0.5 : 0.0;
        tmp = (t == TRAIT_A) ? 1.0 : 0.0;
        if(not ignore_left)
            tmp *= 0.5;
        if(not ignore_right)
            tmp *= 0.5;
        return tmp;
    }
    else if(parent_trait == TRAIT_UU) {
        //return (t == TRAIT_U) ? 0.5 : 0.0;
        tmp = (t == TRAIT_U) ? 1.0 : 0.0;
        if(not ignore_left)
            tmp *= 0.5;
        if(not ignore_right)
            tmp *= 0.5;
        return tmp;
    }
    
    // heterozygotes are informative, so i can look up
    // the recombination fractions
    int p = 0;
    if(parent_trait == TRAIT_UA) {
        p = (t == TRAIT_U) ? 0 : 1;
    }
    else if(parent_trait == TRAIT_AU) {
        p = (t == TRAIT_A) ? 0 : 1;
    }
    
    tmp = 0.5;
    
    if((locus != 0) and (not ignore_left)) {
        tmp *= ((dg->get(person_id, locus-1, parent) == p) ? antitheta2 : theta2);
    }
    
    if((locus != (map->num_markers() - 1)) and (not ignore_right)) {
        tmp *= ((dg->get(person_id, locus+1, parent) == p) ? antitheta : theta);
    }
    
    return tmp;
}
/*
double SamplerRfunction::get_recombination_probability(DescentGraph* dg, unsigned person_id, 
                                                     int maternal_allele, int paternal_allele) {
    double tmp = 1.0;
    
    if((locus != 0) and (not ignore_left)) {
        tmp *= ((dg->get(person_id, locus-1, MATERNAL) == maternal_allele) ? antitheta2 : theta2);
        tmp *= ((dg->get(person_id, locus-1, PATERNAL) == paternal_allele) ? antitheta2 : theta2);
    }
    
    if((locus != (map->num_markers() - 1)) and (not ignore_right)) {
        tmp *= ((dg->get(person_id, locus+1, MATERNAL) == maternal_allele) ? antitheta : theta);
        tmp *= ((dg->get(person_id, locus+1, PATERNAL) == paternal_allele) ? antitheta : theta);
    }
    
    return tmp;
}

double SamplerRfunction::get_transmission_prob(enum phased_trait kid_trait, 
                                                      enum phased_trait parent_trait, 
                                                      enum parentage parent) {
    enum trait t = get_trait(kid_trait, parent);
    
    switch(parent_trait) {
        case TRAIT_AA:
            return (t == TRAIT_A) ? 1.0 : 0.0;
        case TRAIT_UU:
            return (t == TRAIT_U) ? 1.0 : 0.0;
        case TRAIT_AU:
        case TRAIT_UA:
            return 0.5;
    }
    
    abort();
}
*/
void SamplerRfunction::sample(DescentGraph* dg, vector<int>& pmk) {
    double prob_dist[NUM_ALLELES];
    
    // get probabilities
    switch(peel->get_type()) {
        case CHILD_PEEL:
            sample_child(dg, pmk, prob_dist);
            break;
        case LAST_PEEL:
        case PARTNER_PEEL:
            sample_partner(pmk, prob_dist);
            break;
        default:
            fprintf(stderr, "i hav' not written the code for this!\n");
            abort();
    }
        
    // sample
    unsigned int last = 0;
    double r = get_random();
    double total = 0.0;
    
    for(unsigned i = 0; i < NUM_ALLELES; ++i) {
        total += prob_dist[i];
        if(r < total) {
            pmk[peel_id] = i;
            return;
        }
        
        if(prob_dist[i] != 0.0) {
            last = i;
        }
    }
    
    pmk[peel_id] = last;
}

void SamplerRfunction::sample_child(DescentGraph* dg, vector<int>& pmk, double* prob) {
    Person* kid = ped->get_by_index(peel_id);    
    
    enum phased_trait mat_trait;
    enum phased_trait pat_trait;
    enum phased_trait kid_trait;
    
    double tmp;
    double total = 0.0;
    
    mat_trait = static_cast<enum phased_trait>(pmk[kid->get_maternalid()]);
    pat_trait = static_cast<enum phased_trait>(pmk[kid->get_paternalid()]);
    /*
    for(unsigned i = 0; i < NUM_ALLELES; ++i) {
        kid_trait = static_cast<enum phased_trait>(i);
        
        pmk[peel_id] = i;
        
        tmp = get_trait_probability(peel_id, kid_trait);
        
        tmp *= (get_recombination_probability(dg, peel_id, mat_trait, kid_trait, MATERNAL) * \
                get_recombination_probability(dg, peel_id, pat_trait, kid_trait, PATERNAL));
        
        tmp *= ((previous_rfunction1 != NULL) ? previous_rfunction1->get(pmk) : 1.0) * \
               ((previous_rfunction2 != NULL) ? previous_rfunction2->get(pmk) : 1.0);
        
        prob[i] = tmp;
        total += tmp;
    }
    */
    
    for(unsigned i = 0; i < NUM_ALLELES; ++i) {
        kid_trait = static_cast<enum phased_trait>(i);
        
        tmp = (get_recombination_probability(dg, peel_id, mat_trait, kid_trait, MATERNAL) * \
               get_recombination_probability(dg, peel_id, pat_trait, kid_trait, PATERNAL));
        
        total += tmp;
        
        prob[i] = tmp;
    }
    
    if(total == 0.0) {
        fprintf(stderr, "error: probabilities sum to zero (%d)\n", peel_id);
        abort();
    }
    
    for(unsigned i = 0; i < NUM_ALLELES; ++i) {
        prob[i] /= total;
    }
    
    total = 0.0;
    
    for(unsigned i = 0; i < NUM_ALLELES; ++i) {
        kid_trait = static_cast<enum phased_trait>(i);
        
        pmk[peel_id] = i;
        
        tmp = get_trait_probability(peel_id, kid_trait);
        
        /*
        tmp *= (get_recombination_probability(dg, peel_id, mat_trait, kid_trait, MATERNAL) *
                get_recombination_probability(dg, peel_id, pat_trait, kid_trait, PATERNAL));
        */
        tmp *= prob[i];
        
        tmp *= ((previous_rfunction1 != NULL) ? previous_rfunction1->get(pmk) : 1.0) * \
               ((previous_rfunction2 != NULL) ? previous_rfunction2->get(pmk) : 1.0);
        
        prob[i] = tmp;
        total += tmp;
    }
    
    if(total == 0.0) {
        fprintf(stderr, "error: probabilities sum to zero (%d)\n", peel_id);
        abort();
    }
    
    for(unsigned i = 0; i < NUM_ALLELES; ++i) {
        prob[i] /= total;
    }
}

void SamplerRfunction::sample_partner(vector<int>& pmk, double* prob) {
    enum phased_trait partner_trait;    
    double tmp;
    double total = 0.0;
    
    for(unsigned i = 0; i < NUM_ALLELES; ++i) {
        partner_trait = static_cast<enum phased_trait>(i);        
        
        pmk[peel_id] = i;
        
        tmp = get_trait_probability(peel_id, partner_trait);
        
        tmp *= (previous_rfunction1 != NULL ? previous_rfunction1->get(pmk) : 1.0) * \
               (previous_rfunction2 != NULL ? previous_rfunction2->get(pmk) : 1.0);
        
        prob[i] = tmp;
        total += tmp;
    }
    
    if(total == 0.0) {
        fprintf(stderr, "error: probabilities sum to zero (%d)\n", peel_id);
        abort();
    }
    
    for(unsigned i = 0; i < NUM_ALLELES; ++i) {
        prob[i] /= total;
    }
}

void SamplerRfunction::evaluate_child_peel(unsigned int pmatrix_index, DescentGraph* dg) {
        
    unsigned int presum_index;
    Person* kid = ped->get_by_index(peel_id);    
    
    enum phased_trait mat_trait;
    enum phased_trait pat_trait;
    enum phased_trait kid_trait;
    double tmp;
    double total = 0.0;
    
    
    mat_trait = static_cast<enum phased_trait>(indices[pmatrix_index][kid->get_maternalid()]);
    pat_trait = static_cast<enum phased_trait>(indices[pmatrix_index][kid->get_paternalid()]);
    
    
    for(unsigned i = 0; i < NUM_ALLELES; ++i) {
        kid_trait = static_cast<enum phased_trait>(i);
        presum_index = pmatrix_index + (index_offset * i);
        
        indices[pmatrix_index][peel_id] = i;
        
        tmp = get_trait_probability(peel_id, kid_trait);
        if(tmp == 0.0)
            continue;
        
        tmp *= (get_recombination_probability(dg, peel_id, mat_trait, kid_trait, MATERNAL) * \
                get_recombination_probability(dg, peel_id, pat_trait, kid_trait, PATERNAL));
        if(tmp == 0.0)
            continue;
        
        tmp *= ((previous_rfunction1 != NULL) ? previous_rfunction1->get(indices[pmatrix_index]) : 1.0) * \
               ((previous_rfunction2 != NULL) ? previous_rfunction2->get(indices[pmatrix_index]) : 1.0);
        
        pmatrix_presum.set(presum_index, tmp);
        
        total += tmp;
    }
    
    
    /*
    for(int i = 0; i < 2; ++i) {        // maternal
        for(int j = 0; j < 2; ++j) {    // paternal
            
            kid_trait = get_phased_trait(mat_trait, pat_trait, i, j);
            presum_index = pmatrix_index + (index_offset * static_cast<int>(kid_trait));
            
            indices[pmatrix_index][peel_id] = static_cast<int>(kid_trait);
            
            tmp = get_trait_probability(peel_id, kid_trait);
            if(tmp == 0.0)
                continue;
            
            //tmp *= (get_recombination_probability(dg, peel_id, mat_trait, kid_trait, MATERNAL) *
            //        get_recombination_probability(dg, peel_id, pat_trait, kid_trait, PATERNAL));
            
            tmp *= (0.25 * get_recombination_probability(dg, peel_id, i, j));
            
            //tmp *= (get_transmission_prob(kid_trait, mat_trait, MATERNAL) * \
            //        get_transmission_prob(kid_trait, pat_trait, PATERNAL));
            //tmp *= get_recombination_probability(dg, peel_id, i, j);
            if(tmp == 0.0)
                continue;
            
            tmp *= ((previous_rfunction1 != NULL ? previous_rfunction1->get(indices[pmatrix_index]) : 1.0) * \
                    (previous_rfunction2 != NULL ? previous_rfunction2->get(indices[pmatrix_index]) : 1.0));
            
            pmatrix_presum.add(presum_index, tmp);
            
            total += tmp;
        }
    }
    */
    
    pmatrix.set(pmatrix_index, total);
}

void SamplerRfunction::evaluate_parent_peel(unsigned int pmatrix_index, DescentGraph* dg) {
    
    unsigned int presum_index;
    
    enum phased_trait pivot_trait;
    enum phased_trait mat_trait;
    enum phased_trait pat_trait;
    
    double tmp;
    double total = 0.0;
    
    
    for(unsigned int i = 0; i < NUM_ALLELES; ++i) {
        mat_trait = pat_trait = static_cast<enum phased_trait>(i);
        presum_index = pmatrix_index + (index_offset * i);
        
        indices[pmatrix_index][peel_id] = i;
        
        tmp = get_trait_probability(peel_id, mat_trait);
        if(tmp == 0.0)
            continue;
        
        tmp *= ((previous_rfunction1 != NULL ? previous_rfunction1->get(indices[pmatrix_index]) : 1.0) * \
                (previous_rfunction2 != NULL ? previous_rfunction2->get(indices[pmatrix_index]) : 1.0));
        if(tmp == 0.0)
            continue;
        
        double child_prob = 1.0;
        
        for(unsigned int c = 0; c < peel->get_cutset_size(); ++c) {
            unsigned int child_id = peel->get_cutnode(c);
            Person* child = ped->get_by_index(child_id);
            
            if(not child->is_parent(peel_id))
                continue;
            
            
            pivot_trait = static_cast<enum phased_trait>(indices[pmatrix_index][child_id]);
            
            if(child->get_maternalid() == peel_id) {
                pat_trait = static_cast<enum phased_trait>(indices[pmatrix_index][child->get_paternalid()]);
            }
            else {
                mat_trait = static_cast<enum phased_trait>(indices[pmatrix_index][child->get_maternalid()]);
            }
            
            
            child_prob *= (get_recombination_probability(dg, child_id, mat_trait, pivot_trait, MATERNAL) *  \
                           get_recombination_probability(dg, child_id, pat_trait, pivot_trait, PATERNAL));
            
            
            /*
            double child_tmp = 0.0;
            
            for(int i = 0; i < 2; ++i) {        // maternal allele
                for(int j = 0; j < 2; ++j) {    // paternal allele
                    pivot_trait = get_phased_trait(mat_trait, pat_trait, i, j);
                    
                    if(pivot_trait != static_cast<enum phased_trait>(indices[pmatrix_index][child_id]))
                        continue;
                    
                    //child_tmp += (0.25 * get_recombination_probability(dg, peel_id, i, j));
                    
                    child_tmp += (get_recombination_probability(dg, peel_id, mat_trait, pivot_trait, MATERNAL) * \
                                  get_recombination_probability(dg, peel_id, pat_trait, pivot_trait, PATERNAL));
                    
                }
            }
            
            child_prob *= child_tmp;
            */
            
            if(child_prob == 0.0)
                break;
        }
        
        tmp *= child_prob;
        
        pmatrix_presum.set(presum_index, tmp);
        
        total += tmp;
    }
    
    pmatrix.set(pmatrix_index, total);
}

