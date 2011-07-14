#ifndef LKG_SAMPLERRFUNCTION_H_
#define LKG_SAMPLERRFUNCTION_H_

#include <vector>

#include "trait.h"
#include "rfunction.h"
#include "peel_matrix.h"


class DescentGraph;
class Pedigree;
class GeneticMap;

class SamplerRfunction : public Rfunction {
    
    double get_recombination_probability(DescentGraph* dg, 
                                         unsigned locus,
                                         unsigned person_id, 
                                         int maternal_allele, 
                                         int paternal_allele);
    
    double get_trait_probability(unsigned person_id, 
                                 enum phased_trait pt, 
                                 unsigned locus);

 public :
    SamplerRfunction(PeelOperation po, Pedigree* p, GeneticMap* m, Rfunction* prev1, Rfunction* prev2);
    virtual ~SamplerRfunction() {}
    SamplerRfunction(const SamplerRfunction& rhs);
    SamplerRfunction& operator=(const SamplerRfunction& rhs);

    void sample(PeelMatrixKey& pmk);
};

#endif