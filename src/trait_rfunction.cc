using namespace std;

#include "trait_rfunction.h"
#include "rfunction.h"
#include "descent_graph.h"
#include "genetic_map.h"
#include "pedigree.h"
#include "peeling.h"


double TraitRfunction::get_recombination_probability(DescentGraph* dg, unsigned person_id, 
                                                     int maternal_allele, int paternal_allele) {
    double tmp = 1.0;
    
    tmp *= ((dg->get(person_id, locus,   MATERNAL) == maternal_allele) ? antitheta : theta);        
    tmp *= ((dg->get(person_id, locus,   PATERNAL) == paternal_allele) ? antitheta : theta);
    
    tmp *= ((dg->get(person_id, locus+1, MATERNAL) == maternal_allele) ? antitheta2 : theta2);
    tmp *= ((dg->get(person_id, locus+1, PATERNAL) == paternal_allele) ? antitheta2 : theta2);
    
    return tmp;
}
    
double TraitRfunction::get_trait_probability(unsigned person_id, enum phased_trait pt) {
    return (ped->get_by_index(person_id))->get_disease_prob(pt);
}

void TraitRfunction::evaluate_child_peel(unsigned int pmatrix_index, DescentGraph* dg) {
    
    unsigned int presum_index;
    Person* kid = ped->get_by_index(peel_id);
    
    enum phased_trait kid_trait;
    enum phased_trait mat_trait;
    enum phased_trait pat_trait;
    
    double tmp;
    double total = 0.0;

        
    mat_trait = static_cast<enum phased_trait>(indices[pmatrix_index][kid->get_maternalid()]);
    pat_trait = static_cast<enum phased_trait>(indices[pmatrix_index][kid->get_paternalid()]);
    
    // iterate over all descent graphs to determine child trait 
    // based on parents' traits
    for(int i = 0; i < 2; ++i) {        // maternal
        for(int j = 0; j < 2; ++j) {    // paternal
            
            kid_trait = get_phased_trait(mat_trait, pat_trait, i, j);
            presum_index = pmatrix_index + (index_offset * static_cast<int>(kid_trait));
            
            indices[pmatrix_index][peel_id] = static_cast<int>(kid_trait);
            
            tmp = trait_cache[static_cast<int>(kid_trait)];
            if(tmp == 0.0)
                continue;
            
            for(unsigned int k = 0; k < previous_rfunctions.size(); ++k) {
                tmp *= previous_rfunctions[k]->get(indices[pmatrix_index]);
            }
            
            tmp *= (!dg ? 0.25 : 0.25 * get_recombination_probability(dg, peel_id, i, j));
            
            total += tmp;
        }
    }
    
    pmatrix.set(pmatrix_index, total);
}

void TraitRfunction::evaluate_parent_peel(unsigned int pmatrix_index, DescentGraph* dg) {
    
    unsigned int presum_index;
    
    enum phased_trait pivot_trait;
    enum phased_trait mat_trait;
    enum phased_trait pat_trait;
    
    double tmp;
    double total = 0.0;
        
    
    for(unsigned a = 0; a < NUM_ALLELES; ++a) {
        mat_trait = pat_trait = static_cast<enum phased_trait>(a);
        presum_index = pmatrix_index + (index_offset * a);
        
        indices[pmatrix_index][peel_id] = a;
        
        //tmp = get_trait_probability(peel_id, mat_trait);
        tmp = trait_cache[a];
        if(tmp == 0.0)
            continue;
        
        for(unsigned int i = 0; i < previous_rfunctions.size(); ++i) {
            tmp *= previous_rfunctions[i]->get(indices[pmatrix_index]);
        }
        
        double child_prob = 1.0;
        
        for(unsigned c = 0; c < peel->get_cutset_size(); ++c) {
            unsigned int child_id = peel->get_cutnode(c);
            enum phased_trait kid_trait = static_cast<enum phased_trait>(indices[pmatrix_index][child_id]);
            
            Person* child = ped->get_by_index(child_id);
            
            if(not child->is_parent(peel_id))
                continue;
            
            if(child->get_maternalid() == peel_id) {
                pat_trait = static_cast<enum phased_trait>(indices[pmatrix_index][child->get_paternalid()]);
            }
            else {
                mat_trait = static_cast<enum phased_trait>(indices[pmatrix_index][child->get_maternalid()]);
            }
            
            double child_tmp = 0.0;
            
            for(int i = 0; i < 2; ++i) {        // maternal allele
                for(int j = 0; j < 2; ++j) {    // paternal allele
                    pivot_trait = get_phased_trait(mat_trait, pat_trait, i, j);
                    
                    if(pivot_trait != kid_trait)
                        continue;
                    
                    child_tmp += (!dg ? 0.25 : 0.25 * get_recombination_probability(dg, child_id, i, j));
                }
            }
            
            child_prob *= child_tmp;
        }
        
        tmp *= child_prob;
        
        total += tmp;
    }
    
    pmatrix.set(pmatrix_index, total);
}

