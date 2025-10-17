#include <vector>
#include <cstring>
#include <cuda_runtime.h>

#include "kseq/kseq.h"
#include "common.h"
#include "device_seq_t.h"

#define BLOCK_SIZE 256

__device__ int d_match_results_count = 0;

__global__ void myKernel(const device_seq_t* d_samples, int num_samples, const device_seq_t* d_signatures, 
                int num_signatures, device_match_result_t* match_results) {

        int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

        const int sample_idx    = blockIdx.x;
        const int signature_idx = blockIdx.y;
        const int tid           = threadIdx.x;

        const device_seq_t& sample    = d_samples[sample_idx];
        const device_seq_t& signature = d_signatures[signature_idx];

        int start = tid;
        int end = sample.seq_len - signature.seq_len;
        if (end < 0) return;

        long long unsigned checksum = 0;
        int best_sum = 0;
        // thread local match result
        device_match_result_t match_result {};
        match_result.sample_name = sample.name;
        match_result.signature_name = signature.name;
        match_result.match_score = 0;

        for (int i = start; i < end; i += blockDim.x) {
                int curr_sum = 0;
                for (int j = 0; j < signature.seq_len; ++j) {
                        char t = signature.seq[j];
                        char s = sample.seq[i + j];

                        if (s == 'N' || t == 'N') continue;

                        if (s == t) {
                                curr_sum += static_cast<int>(sample.qual[i + j] - 33);
                        } else {
                                curr_sum = 0;
                                break;
                        } 
                }
                if (curr_sum > best_sum) {
                        // better match found
                        best_sum = curr_sum;
                }
        }
        match_result.match_score = best_sum / static_cast<double>(signature.seq_len);
        match_result.integrity_hash = checksum % 97;

        // copy 
        __shared__ double block_scores[BLOCK_SIZE];
        block_scores[tid] = best_sum;
        __syncthreads();

        // reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride && block_scores[tid + stride] > block_scores[tid]) {
                        block_scores[tid] = block_scores[tid + stride];
                }
                __syncthreads();
        }

        // write to device match results
        if (tid == 0) {
                int idx = atomicAdd(&d_match_results_count, 1);
                match_results[idx].sample_name = sample.name;
                match_results[idx].signature_name = signature.name;
                match_results[idx].match_score = block_scores[0];
        }

}

void runMatcher(const std::vector<klibpp::KSeq>& samples,
                const std::vector<klibpp::KSeq>& signatures,
                std::vector<MatchResult>& match_results) {

        // -------------------------------------------------------------------------
        // Allocate arrays of structs (managed so both CPU and GPU can access)
        // -------------------------------------------------------------------------
        device_seq_t* d_samples = nullptr;
        device_seq_t* d_signatures = nullptr;

        cudaMallocManaged(&d_samples, samples.size() * sizeof(device_seq_t));
        cudaMallocManaged(&d_signatures, signatures.size() * sizeof(device_seq_t));

        // -------------------------------------------------------------------------
        // Copy sample data
        // -------------------------------------------------------------------------
        for (size_t i = 0; i < samples.size(); ++i) {
                const auto& s = samples[i];
                d_samples[i].seq_len = s.seq.size();

                cudaMallocManaged(&d_samples[i].name, s.name.size() + 1);
                cudaMallocManaged(&d_samples[i].seq,  s.seq.size()  + 1);
                cudaMallocManaged(&d_samples[i].qual, s.qual.size() + 1);

                std::strcpy(d_samples[i].name, s.name.c_str());
                std::strcpy(d_samples[i].seq,  s.seq.c_str());
                std::strcpy(d_samples[i].qual, s.qual.c_str());
        }

        // -------------------------------------------------------------------------
        // Copy signature data
        // -------------------------------------------------------------------------
        for (size_t i = 0; i < signatures.size(); ++i) {
                const auto& sig = signatures[i];
                d_signatures[i].seq_len = sig.seq.size();

                cudaMallocManaged(&d_signatures[i].name, sig.name.size() + 1);
                cudaMallocManaged(&d_signatures[i].seq,  sig.seq.size()  + 1);
                cudaMallocManaged(&d_signatures[i].qual, sig.qual.size() + 1);

                std::strcpy(d_signatures[i].name, sig.name.c_str());
                std::strcpy(d_signatures[i].seq,  sig.seq.c_str());
                std::strcpy(d_signatures[i].qual, sig.qual.c_str());
        }

        // -------------------------------------------------------------------------
        // Allocate space for match results
        // -------------------------------------------------------------------------

        // At most 1% of samples have a virus
        int match_results_size = static_cast<int>(ceil(samples.size() / 100.0));

        // Allocate array of structs in unified memory
        device_match_result_t* device_match_results = nullptr;
        cudaMallocManaged(&device_match_results, match_results_size * sizeof(device_match_result_t));

        // Reasonable maximum name lengths
        const int MAX_SAMPLE_NAME_LEN    = 50;
        const int MAX_SIGNATURE_NAME_LEN = 50;

        // Allocate name buffers for each result
        for (int i = 0; i < match_results_size; ++i) {
                cudaMallocManaged(&device_match_results[i].sample_name,    MAX_SAMPLE_NAME_LEN);
                cudaMallocManaged(&device_match_results[i].signature_name, MAX_SIGNATURE_NAME_LEN);

                // Optional init
                device_match_results[i].sample_name[0]    = '\0';
                device_match_results[i].signature_name[0] = '\0';
                device_match_results[i].match_score = 0.0;
                device_match_results[i].integrity_hash = 0;
        }
        // reset global counter
        cudaMemset(&d_match_results_count, 0, sizeof(int));

        // -------------------------------------------------------------------------
        // Launch kernel 
        // -------------------------------------------------------------------------
        dim3 gridDim(samples.size(), signatures.size()); 
        dim3 blockDim(BLOCK_SIZE, 1, 1);

        myKernel<<<gridDim, blockDim>>>(d_samples, samples.size(), d_signatures, signatures.size(), device_match_results);

        cudaDeviceSynchronize();

        // -------------------------------------------------------------------------
        // Process Match Results
        // -------------------------------------------------------------------------
        for (int i = 0; i < match_results_size; ++i) {
                MatchResult res;

                // Copy elements from device memory to host std::string
                res.sample_name    = std::string(device_match_results[i].sample_name);
                res.signature_name = std::string(device_match_results[i].signature_name);
                res.match_score    = device_match_results[i].match_score;
                res.integrity_hash = device_match_results[i].integrity_hash;

                // Push into vector
                match_results.push_back(std::move(res));
        }

        // -------------------------------------------------------------------------
        // Cleanup
        // -------------------------------------------------------------------------
        for (size_t i = 0; i < samples.size(); ++i) {
                cudaFree(d_samples[i].name);
                cudaFree(d_samples[i].seq);
                cudaFree(d_samples[i].qual);
        }

        for (size_t i = 0; i < signatures.size(); ++i) {
                cudaFree(d_signatures[i].name);
                cudaFree(d_signatures[i].seq);
                cudaFree(d_signatures[i].qual);
        }

        cudaFree(d_samples);
        cudaFree(d_signatures);
}

