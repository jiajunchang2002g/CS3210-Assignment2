const int BLOCK_SIZE = 256; 
// -----------------------------------------------------------------------------
// Device structs
// -----------------------------------------------------------------------------
struct d_KSeq {
        const char* name;
        const char* seq;
        const char* qual;
        int seq_len;
};

struct d_MatchResult {
        const char* sample_name;
        const char* signature_name;
        double match_score;
        int integrity_hash;
};

// -----------------------------------------------------------------------------
// Helper to convert struct to d_KSeq
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Host structs
// -----------------------------------------------------------------------------
struct KSeq { // kseq_t
        std::string name;
        std::string comment;
        std::string seq;
        std::string qual;
}

struct MatchResult {
        std::string sample_name;
        std::string signature_name;
        double match_score;
        int integrity_hash;
};

// -----------------------------------------------------------------------------
// GPU kernel
// -----------------------------------------------------------------------------
__global__ void strstr(const d_KSeq* samples,
                const d_KSeq* signatures,
                d_MatchResult* results,
                int num_samples,
                int num_signatures)
{
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        int tid = threadIdx.x;

        int total = num_samples * num_signatures;
        if (id >= total) return;

        __shared__ MatchResult partial_results[BLOCK_SIZE];

        // TODO: map block_id to sig-samp pair, map thread to its portion of samp
        int sample_id;
        int sig_id;
        const KSeq& sample = samples[sample_id];
        const KSeq& signature = signatures[sig_id];

        int start_index; 
        int end_index;

        // sum of phred scores of entire sequeunce
        int total_sum = 0;
        // sum of phred scores of matched portion
        int best_sum = 0;

        MatchResult best_match;
        best_match.sample_name = sample.name;
        best_match.signature_name = signature.name;

        for (int i = start_index; i <= end_index; ++i) {
                int curr_sum = 0;

                // maybe abstract out as a function isMatching(partial_samp, sig);
                for (int j = 0; j < signature.seq_len; ++j) {
                        char s = sample.seq[i + j];
                        char t = signature.seq[j];
                        sum += static_cast<int>(sample.qual[i + j] - 33);

                        if (s == 'N' || t == 'N') continue;

                        if (s == t)
                                curr_sum += static_cast<int>(sample.qual[i + j] - 33);
                        else {
                                curr_sum = 0;
                                break;
                        }
                }

                if (curr_sum > best_sum) {
                        best_sum = curr_sum;
                        result.match_score = best_sum / static_cast<double>(signature.seq_len);
                        result.integrity_hash = sum % 97;
                }
        }

        partial_results[tid] = result;
        // parallel reduce on partial_results to obtain best match_result based on match_score
        // copy to global memory
}

void runMatcher(const std::vector<klibpp::KSeq> &samples,
                const std::vector<klibpp::KSeq> &signatures,
                std::vector<MatchResult> &matches);
int main() {
        // preproces vector of strings into array of simple char* struct
        // cudaAlloc and memcpy samples and signatures to device global mem
        // cudaAlloc for results, 1% * number samples * sizeof(MatchResult), pass in this ptr
        // invoke kernel (h_result, signatures, samples)
        // convert h_result to vector of string struct
        strstr();
}
