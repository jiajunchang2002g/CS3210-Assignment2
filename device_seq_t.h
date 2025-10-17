// Number of signatures: [500,1000]
// Number of samples: [1000,2200]
// Sample DNA Length: [100000,2000000]
// Signature DNA Length: [3000,10000]
// Probability that any given nucleotide is a specific DNA sequence is ‘N’: [0,0.1]
// Maximum percentage of samples with (any no. of) viruses: 1%

struct device_seq_t {
        char *name;
        char *seq;
        char *qual;
        int seq_len;
};

struct device_match_result_t {
        char *sample_name;
        char *signature_name;
        double match_score;
        int integrity_hash;
};

