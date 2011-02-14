#ifndef LKG_PEDIGREEPARSER_H_
#define LKG_PEDIGREEPARSER_H_

using namespace std;

#include <string>
#include <vector>

#include "genotype.h"
#include "person.h"
#include "parser.h"


class Pedigree;

class PedigreeParser : public Parser {

	vector<Pedigree>& pedigrees;
	
	bool _parse_sex(const string& str, enum sex& s);
	bool _parse_affection(const string& str, enum affection& a);
	bool _parse_genotype(const string& a1, const string& a2, enum unphased_genotype& g);
	Pedigree& _current_ped(const string& famid);
	
 public :
	PedigreeParser(const string fn, vector<Pedigree>& peds) 
		: Parser(fn, false), pedigrees(peds) {}
    
	bool parse_line(const int linenum, const string line);
    bool parse_end();
};

#endif

