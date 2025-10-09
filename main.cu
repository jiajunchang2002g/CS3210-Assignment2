const int BLOCK_SIZE = 256; 
// -----------------------------------------------------------------------------
// GPU-safe representations
// -----------------------------------------------------------------------------
struct KSeq {
        const char* name;
        const char* seq;
        const char* qual;
        int seq_len;
};

struct MatchResult {
        const char* sample_name;
        const char* signature_name;
        double match_score;
        int integrity_hash;
};

// -----------------------------------------------------------------------------
// GPU kernel
// -----------------------------------------------------------------------------
__global__ void strstr(const KSeq* samples,
                const KSeq* signatures,
                DeviceMatchResult* results,
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
        // use reduction to grab best result from partial_results

}

int main() {
        // init some signatures and samples
        KSeq[] signatures;
        KSeq[] samples;
        // call strstr
        strstr();
}
