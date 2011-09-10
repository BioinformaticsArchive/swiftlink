#ifndef LKG_PEELMATRIX_H_
#define LKG_PEELMATRIX_H_

using namespace std;

#include <cstdio>
#include <cmath>
#include <map>
#include <vector>
#include <algorithm>
#include <string>

#include "trait.h"


class PeelMatrixKey {

    //map<unsigned int, enum phased_trait> key;
    //vector<enum phased_trait> key;
    unsigned int num_keys;
    enum phased_trait* key;
    
 public :
    PeelMatrixKey(unsigned max_keys) :
        num_keys(max_keys), 
        key(NULL) {
    
        key = new enum phased_trait[num_keys];
        for(unsigned i = 0; i < num_keys; ++i)
            key[i] = TRAIT_UU;
    }
/*    
    PeelMatrixKey(vector<unsigned int>& cutset, vector<unsigned int>& assignments) : key() {
        reassign(cutset, assignments);
    }
*/
    ~PeelMatrixKey() {
        delete[] key;
    }

    PeelMatrixKey(const PeelMatrixKey& rhs) :
        num_keys(rhs.num_keys), 
        key(NULL) {
        
        key = new enum phased_trait[num_keys];
        copy(rhs.key, rhs.key + num_keys, key);
    }

    PeelMatrixKey& operator=(const PeelMatrixKey& rhs) {
        if(this != &rhs) {
            if(rhs.num_keys != num_keys) {
                num_keys = rhs.num_keys;
                delete[] key;
                key = new enum phased_trait[num_keys];
            }
            
            copy(rhs.key, rhs.key + num_keys, key);
        }

        return *this;
    }
    
    void reassign(vector<unsigned int>& cutset, vector<unsigned int>& assignments) {
        for(unsigned int i = 0; i < cutset.size(); ++i) {
            add(
                cutset[i], 
                static_cast<enum phased_trait>(assignments[i])
            );
        }
    }

    void add(unsigned int k, enum phased_trait value) {
        key[k] = value;
    }
/*
    void remove(unsigned int k) {
        key.erase(k);
    }
*/
    inline enum phased_trait get(unsigned int i) {
        return key[i];
    }
/*    
    // ensure this key can address everything for everything in the
    // vector 'keys'
    bool check_keys(vector<unsigned int>& keys) {
        
        if(keys.size() != key.size()) {
            return false;
        }

        for(unsigned int i = 0; i < keys.size(); ++i) {
            if(key.count(keys[i]) == 0) {
                return false;
            }
        }

        return true;
    }
*/
    void print() {
/*        map<unsigned int, enum phased_trait>::iterator it;
        
        for(it = key.begin(); it != key.end(); it++) {
            printf("%d=%d ", (*it).first, (*it).second);
        }
*/
    }
    
    void raw_print() {
        for(unsigned int i = 0; i < num_keys; ++i)
            printf("%d ", (int) key[i]);
        printf("\n");
    }
};

class PeelMatrix {

    //vector<unsigned int> keys;
    //vector<unsigned int> offsets;
    unsigned int num_keys;
    unsigned int* keys;
    unsigned int* offsets;
    unsigned int number_of_dimensions;
    unsigned int values_per_dimension;
    unsigned int size;
    double* data;
    
    //unsigned int generate_index(PeelMatrixKey& pmk) const;
    inline unsigned int generate_index(PeelMatrixKey& pmk) const {
        unsigned int index = 0;

        for(unsigned int i = 0; i < num_keys; ++i) {
            //index += (offsets[i] * pmk.get(keys[i]));
            index |= (pmk.get(keys[i]) << (i * 2));
        }
        
        return index;
    }
    
    void init_offsets();
    
 public :
    PeelMatrix(unsigned int num_dim, unsigned int val_dim);
    PeelMatrix(const PeelMatrix& rhs);
    PeelMatrix& operator=(const PeelMatrix& rhs);
    ~PeelMatrix();

    /*
    bool key_intersection(
            PeelMatrix* pm, 
            vector<unsigned int>& missing, 
            vector<unsigned int>& additional
        );
    */
    
    void set_keys(vector<unsigned int>& k);
    //bool is_legal(PeelMatrixKey& pmk);
    
    //double get(PeelMatrixKey& pmk) const;
    //void set(PeelMatrixKey& pmk, double value);
    //void add(PeelMatrixKey& pmk, double value);
    double get_result();

    double sum();
    void normalise();
    
    //void generate_key(PeelMatrixKey& pmatrix_index, vector<unsigned int>& assignments);
    //void print();
    //void print_keys();
    
    inline double get(PeelMatrixKey& pmk) const {
        return data[generate_index(pmk)];
    }
    
    inline void set(PeelMatrixKey& pmk, double value) {
        if(value != 0.0)
            data[generate_index(pmk)] = value;
    }
    
    inline void add(PeelMatrixKey& pmk, double value) {
        if(value != 0.0)
            data[generate_index(pmk)] += value;
    }
    
    void reset();
    
    void raw_print() {
        for(unsigned int i = 0; i < size; ++i) {
            printf("%.3f\n", data[i]);
        }
        printf("\n");
    }
};

#endif

