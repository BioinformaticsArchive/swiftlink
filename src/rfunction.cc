using namespace std;

#include <cmath>
#include <vector>

#include "rfunction.h"
#include "peeling.h"
#include "peel_matrix.h"
#include "genotype.h"
#include "pedigree.h"
#include "descent_graph.h"
#include "trait.h"
#include "genetic_map.h"


Rfunction::Rfunction(Pedigree* p, GeneticMap* m, unsigned int locus, PeelOperation* po, Rfunction* prev1, Rfunction* prev2) : 
    map(m),
    ped(p),
    offset(0.0),
    pmatrix(po->get_cutset_size(), NUM_ALLELES),
    pmatrix_presum(po->get_cutset_size() + 1, NUM_ALLELES),
    peel(po), 
    previous_rfunction1(prev1),
    previous_rfunction2(prev2),
    locus(locus),
    indices(peel->get_index_values()),
    index_offset(1 << (2 * peel->get_cutset_size())),
    size(pow((double)NUM_ALLELES, (int)peel->get_cutset_size())),
    peel_id(peel->get_peelnode()) {
    
    pmatrix.set_keys(peel->get_cutset());
          
    // XXX temporary, neater way of doing this?
    vector<unsigned> tmp(peel->get_cutset());
    tmp.push_back(peel->get_peelnode());
    
    pmatrix_presum.set_keys(tmp);       
}

Rfunction::Rfunction(const Rfunction& rhs) :
    map(rhs.map),
    ped(rhs.ped),
    offset(rhs.offset),
    pmatrix(rhs.pmatrix),
    pmatrix_presum(rhs.pmatrix_presum),
    peel(rhs.peel),
    previous_rfunction1(rhs.previous_rfunction1),
    previous_rfunction2(rhs.previous_rfunction2),
    locus(rhs.locus),
    indices(rhs.indices),
    index_offset(rhs.index_offset),
    size(rhs.size),
    peel_id(rhs.peel_id) {}
    
Rfunction& Rfunction::operator=(const Rfunction& rhs) {

    if(&rhs != this) {
        pmatrix = rhs.pmatrix;
        pmatrix_presum = rhs.pmatrix_presum;
        peel = rhs.peel;
        map = rhs.map;
        ped = rhs.ped;
        offset = rhs.offset;
        previous_rfunction1 = rhs.previous_rfunction1;
        previous_rfunction2 = rhs.previous_rfunction2;
        locus = rhs.locus;
        indices = rhs.indices;
        index_offset = rhs.index_offset;
        size = rhs.size;
        peel_id = rhs.peel_id;
    }
    
    return *this;
}

bool Rfunction::affected_trait(enum phased_trait pt, int allele) {
    
    switch(allele) {
        case 0 :
            return (pt == TRAIT_AU) or (pt == TRAIT_AA);
        case 1 :
            return (pt == TRAIT_UA) or (pt == TRAIT_AA);
        default :
            break;
    }
    
    abort();
}

enum phased_trait Rfunction::get_phased_trait(enum phased_trait m, enum phased_trait p, 
                                                   int maternal_allele, int paternal_allele) {
                                                   
    bool m_affected = affected_trait(m, maternal_allele);
    bool p_affected = affected_trait(p, paternal_allele);
    enum phased_trait pt;
    
    if(m_affected) {
        pt = p_affected ? TRAIT_AA : TRAIT_AU;
    }
    else {
        pt = p_affected ? TRAIT_UA : TRAIT_UU;
    }
    
    return pt;
}

// this function is the same for Traits and Sampling
void Rfunction::evaluate_partner_peel(unsigned int pmatrix_index) {
    double tmp = 0.0;
    double total = 0.0;
    
    unsigned int presum_index;
    
    enum phased_trait partner_trait;
    
    
    for(unsigned i = 0; i < NUM_ALLELES; ++i) {
        partner_trait = static_cast<enum phased_trait>(i);        
        presum_index = pmatrix_index + (index_offset * i);
        
        indices[pmatrix_index][peel_id] = i;
        
        tmp = get_trait_probability(peel_id, partner_trait);
        if(tmp == 0.0)
            continue;
        
        tmp *= (previous_rfunction1 == NULL ? 1.0 : previous_rfunction1->get(indices[pmatrix_index])) * \
               (previous_rfunction2 == NULL ? 1.0 : previous_rfunction2->get(indices[pmatrix_index]));
        
        pmatrix_presum.set(presum_index, tmp);
        
        total += tmp;
    }
    
    pmatrix.set(pmatrix_index, total);
}

void Rfunction::evaluate_element(unsigned int pmatrix_index, DescentGraph* dg) {
    // XXX could remove this with some inheritance?
    // RfunctionChild RfunctionParent?
    switch(peel->get_type()) {
        
        case CHILD_PEEL :
            evaluate_child_peel(pmatrix_index, dg);
            break;
            
        case PARTNER_PEEL :
        case LAST_PEEL :
            evaluate_partner_peel(pmatrix_index);
            break;
        
        case PARENT_PEEL :
            evaluate_parent_peel(pmatrix_index, dg);
            break;
        
        default :
            fprintf(stderr, "error: default should never be reached! (%s:%d)\n", __FILE__, __LINE__);
            abort();
    }
}

bool Rfunction::legal_genotype(unsigned personid, enum phased_trait g) {
    Person* p = ped->get_by_index(personid);
    
    return p->legal_genotype(locus, g);
}

void Rfunction::evaluate(DescentGraph* dg, double offset) {
    pmatrix.reset();
    pmatrix_presum.reset();
    
    preevaluate_init(dg);
    
    // crucial for TraitRfunction
    this->offset = offset;
    
    //#pragma omp parallel for
    for(unsigned int i = 0; i < size; ++i) {
        evaluate_element(i, dg);
    }
}

void Rfunction::normalise(double* p) {
    double total = p[0] + p[1] + p[2] + p[3];
    
    if(total == 0.0)
        return;
    
    for(int i = 0; i < 4; ++i) {
        p[i] /= total;
    }
}

enum trait Rfunction::get_trait(enum phased_trait p, enum parentage parent) {
    switch(parent) {
        case MATERNAL:
            return (((p == TRAIT_UU) or (p == TRAIT_UA)) ? TRAIT_U : TRAIT_A);    
        case PATERNAL:
            return (((p == TRAIT_UU) or (p == TRAIT_AU)) ? TRAIT_U : TRAIT_A);
        default:
            break;
    }
    
    abort();
}

