using namespace std;

#include <cmath>

#include "rfunction.h"
#include "peeling.h"
#include "peel_matrix.h"
#include "genotype.h"
#include "pedigree.h"
#include "simwalk_descent_graph.h"
#include "trait.h"
#include "genetic_map.h"


Rfunction::Rfunction(PeelOperation po, Pedigree* p, GeneticMap* m, unsigned int alleles)
    : pmatrix(po.get_cutset_size(), alleles), 
      peel(po), 
      num_alleles(alleles), 
      map(m),
      ped(p) {
        
    if(alleles != 4)
        abort(); // XXX assumption for now...

    pivot = ped->get_by_index(peel.get_pivot());
    pmatrix.set_keys(peel.get_cutset());
}

void Rfunction::generate_key(PeelMatrixKey& pmatrix_index, vector<unsigned int>& assignments) {
    pmatrix_index.reassign(peel.get_cutset(), assignments);
}

bool Rfunction::affected_trait(enum phased_trait pt, int allele) {
    
    switch(allele) {
        case 0 :
            return (pt == TRAIT_AU) or (pt == TRAIT_AA) ? true : false;

        case 1 :
            return (pt == TRAIT_UA) or (pt == TRAIT_AA) ? true : false;
        
        default :
            abort();
    }

    return false;
}

double Rfunction::get_disease_probability(
                    enum phased_trait m, enum phased_trait p, 
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

    return pivot->get_disease_prob(pt);
}

// XXX this needs to be somewhere that is not at the same position as 
// a genetic marker
// TODO XXX make this generic so it can do arbitrary points between markers
double Rfunction::get_recombination_probability(
                    SimwalkDescentGraph* dg, unsigned int locus_index,
                    int maternal_allele, int paternal_allele) {

    double tmp = 1.0;
    double half_recomb_prob;
    
    half_recomb_prob = map->get_theta_halfway(locus_index);
    
    tmp *= dg->get(pivot->get_internalid(), locus_index,   MATERNAL) == maternal_allele ? \
            1.0 - half_recomb_prob : half_recomb_prob;
    tmp *= dg->get(pivot->get_internalid(), locus_index+1, MATERNAL) == maternal_allele ? \
            1.0 - half_recomb_prob : half_recomb_prob;
    tmp *= dg->get(pivot->get_internalid(), locus_index,   PATERNAL) == paternal_allele ? \
            1.0 - half_recomb_prob : half_recomb_prob;
    tmp *= dg->get(pivot->get_internalid(), locus_index+1, PATERNAL) == paternal_allele ? \
            1.0 - half_recomb_prob : half_recomb_prob;
    
    return tmp;
}

void Rfunction::evaluate_child_peel(
                    PeelMatrixKey& pmatrix_index, 
                    PeelMatrix* prev_matrix, 
                    SimwalkDescentGraph* dg,
                    unsigned int locus_index) {

    double tmp;
    double recombination_prob;
    double disease_prob;
    double old_prob;
    enum phased_trait mat_trait;
    enum phased_trait pat_trait;
    PeelMatrixKey prev_index(pmatrix_index);
    vector<unsigned int> missing;
    vector<unsigned int> additional;


    if(prev_matrix and not pmatrix.key_intersection(prev_matrix, missing, additional)) {
        // keys are the same, ie: when peeling sibs in the same nuclear
        // family
        
        // get rid of the additional things
        //  - should be the parents
        if(additional.size() != 2) {
            fprintf(stderr, "too many people to remove! (%s %d)\n", __FILE__, __LINE__);
            for(unsigned int i = 0; i < additional.size(); ++i) {
                fprintf(stderr, "additional[%d] = %d\n", i, additional[i]);
            }
            abort();
        }

        prev_index.remove(additional[0]);
        prev_index.remove(additional[1]);

        // add the missing things
        //  - should only be the pivot? 
        //      * true: in a simple peel
        //      * false: more in a complex one, but only the pivot will
        //               need to be added because the rest will be in the 
        //               pmatrix_index
        // XXX only simple peeling sequences for now...
        if(missing.size() != 1) {
            fprintf(stderr, "too many people to add! (%s %d)\n", __FILE__, __LINE__);
            for(unsigned int i = 0; i < missing.size(); ++i) {
                fprintf(stderr, "missing[%d] = %d\n", i, missing[i]);
            }
            pmatrix_index.print();
            abort();
        }
    }

//    fprintf(stderr, "%d missing people\n%d additional people\n",
//        missing.size(), additional.size());
    
//    // XXX this will not always work
//    // need to differentiate from the first child in a marriage being
//    // peeled and every subsequent child
//    prev_index.remove(pivot->get_maternalid()); // XXX this work should be cached(?)
//    prev_index.remove(pivot->get_paternalid()); // XXX this work should be cached(?)

    mat_trait = pmatrix_index.get(pivot->get_maternalid());
    pat_trait = pmatrix_index.get(pivot->get_paternalid());
    
    // iterate over all descent graphs to determine child trait based on 
    // parents' traits
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 2; ++j) {
            
            disease_prob        = get_disease_probability(mat_trait, pat_trait, i, j);
            recombination_prob  = get_recombination_probability(dg, locus_index, i, j);
            old_prob            = prev_matrix != NULL ? prev_matrix->get(prev_index) : 1.0;

            tmp += (disease_prob * recombination_prob * old_prob);
        }
    }

    pmatrix.set(pmatrix_index, tmp);

    pmatrix_index.print();
    printf(" := %f\n", tmp);
}

void Rfunction::evaluate_partner_peel(
                    PeelMatrixKey& pmatrix_index, 
                    PeelMatrix* prev_matrix, 
                    SimwalkDescentGraph* dg,
                    unsigned int locus_index) {
    
    double tmp = 0.0;

    // TODO
    
    pmatrix_index.print();
    printf(" := %f\n", tmp);
}

void Rfunction::evaluate_element(
                    PeelMatrixKey& pmatrix_index, 
                    PeelMatrix* prev_matrix, 
                    SimwalkDescentGraph* dg, 
                    unsigned int locus_index) {
    
    // given that 'prev_matrix' exists, we need to be able to query it
    // how this is performed depends on the 'type' of peel we are talking
    // about and I am not sure whether this procedure is (or can be) particularly
    // general
    //
    // XXX perhaps the PeelMatrixKey class should have the responsibility of 
    // figuring this out?
    //
    // XXX this could all be sped up with template probably (?)
    switch(peel.get_type()) {
        case CHILD_PEEL :
            printf("child\n");
            evaluate_child_peel(pmatrix_index, prev_matrix, dg, locus_index);
            break;
            
        case PARTNER_PEEL :
            printf("partner\n");
            evaluate_partner_peel(pmatrix_index, prev_matrix, dg, locus_index);
            break;
        
        case PARENT_PEEL :  // XXX don't bother with yet
        case LAST_PEEL :    // XXX never seen here? just a sum, handle in 'Rfunction::evaluate'
            printf("other\n");
        default :
            abort();
    }
}

// XXX can i tell if these matrix can be used together
//
bool Rfunction::evaluate(PeelMatrix* previous_matrix, SimwalkDescentGraph* dg, unsigned int locus_index) {
    PeelMatrixKey k;
    vector<unsigned int> q;
    unsigned int ndim = peel.get_cutset_size();
    unsigned int tmp;
    unsigned int i;
        
    // initialise to the first element of matrix
    for(i = 0; i < ndim; ++i) {
        q.push_back(0);
    }

    // enumerate all elements in ndim-dimenstional matrix
    while(not q.empty()) {
        
        if(q.size() == ndim) {
            generate_key(k, q);
            evaluate_element(k, previous_matrix, dg, locus_index);
        }
        
        tmp = q.back() + 1;
        q.pop_back();
        
        if(tmp < num_alleles) {
            q.push_back(tmp);
            tmp = ndim - q.size();
            // fill out rest with zeroes
            for(i = 0; i < tmp; ++i) {
                q.push_back(0);
            }
        }
    }

    return true;
}

