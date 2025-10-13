#include <vector>
#include <cstring>
#include <cuda_runtime.h>

#include "kseq/kseq.h"
#include "common.h"
#include "device_seq_t.h"

/*

struct device_match_result_t {
        char *sample_name;
        char *signature_name;
        double match_score;
        int integrity_hash;
}

*/

void myKernel(device_seq_t* d_samples, samples.size(), d_signatures, signatures.size(), match_results.data()) {

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

        int match_results_size = 0.01 * samples.size();
        device_match_result_t device_match_results;

        // -------------------------------------------------------------------------
        // Launch your matcher kernel (example)
        // -------------------------------------------------------------------------
        dim3 gridDim();
        dim3 blockDim(256, 1, 1);

        myKernel<<<gridDim, blockDim>>>(d_samples, samples.size(), d_signatures, signatures.size(), match_results.data());

        cudaDeviceSynchronize();

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

