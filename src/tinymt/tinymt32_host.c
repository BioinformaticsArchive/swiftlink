/**
 * @file tinymt32_host.c
 *
 * @brief utility program for CUDA implementation of TinyMT32.
 *
 * This is utility proogram for CUDA implementation of TinyMT32.
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (The University of Tokyo)
 *
 * Copyright (C) 2011 Mutsuo Saito, Makoto Matsumoto,
 * Hiroshima University and The University of Tokyo.
 * All rights reserved.
 *
 * The 3-clause BSD License is applied to this software, see LICENSE.txt
 */

#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "tinymt32_host.h"

static int read_line32(uint32_t *mat1, uint32_t *mat2,
		       uint32_t *tmat, FILE *fp);


/**
 * This function reads parameter from file and puts them in an array.
 * The file should be the output of tinymt32dc.
 *
 * @param filename name of the file generated by tinymt32dc.
 * @param params output array of this function.
 * number of elements of the array is num_params * 3.
 * @param num_param number of parameters
 * @return 0 if normal end.
 */
int tinymt32_set_params(const char * filename,
			uint32_t * params,
			int num_param)
{
    FILE *ifp;
    int rc;
    uint32_t mat1 = 0;
    uint32_t mat2 = 0;
    uint32_t tmat = 0;
    int i;
    
    ifp = fopen(filename, "r");
    if (ifp == NULL) {
	    return -1;
    }
    
    for (i = 0; i < num_param; i++) {
	    rc = read_line32(&mat1, &mat2, &tmat, ifp);
	    
	    if (rc != 0) {
	        return -2;
	    }
	
	    params[i * 3 + 0] = mat1;
	    params[i * 3 + 1] = mat2;
	    params[i * 3 + 2] = tmat;
    }
    
    fclose(ifp);
    
    return 0;
}

/**
 * read line from fp and set parametes to mat1, mat2, tmat.
 * The format of the file should be that of tinymt32dc's output.
 *
 * @param mat1 output mat1 parameter.
 * @param mat2 output mat2 parameter.
 * @param tmat output tmat parameter.
 * @param fp file pointer.
 * @return 0 if normal end.
 */
static int read_line32(uint32_t *mat1, uint32_t *mat2, uint32_t *tmat, FILE *fp)
{
#define BUFF_SIZE 500
    char buff[BUFF_SIZE];
    char * p;
    uint32_t num;
    int i;
    
    errno = 0;
    for (;;) {
	    if (feof(fp) || ferror(fp)) {
	        return -1;
	    }
	    
	    if(fgets(buff, BUFF_SIZE, fp) == NULL) {
	    //if (errno) {
	        return errno;
	    }
	    
	    if (buff[0] != '#') {
	        break;
	    }
    }
    
    p = buff;
    
    for (i = 0; i < 3; i++) {
	    p = strchr(p, ',');
	    if (p == NULL) {
	        return -1;
	    }
	    p++;
    }
    
    num = strtoul(p, &p, 16);
    if (errno) {
	    return errno;
    }
    *mat1 = num;
    p++;
    
    num = strtoul(p, &p, 16);
    if (errno) {
	    return errno;
    }
    *mat2 = num;
    p++;
    
    num = strtoul(p, &p, 16);
    if (errno) {
	    return errno;
    }
    *tmat = num;
    
    return 0;
}

