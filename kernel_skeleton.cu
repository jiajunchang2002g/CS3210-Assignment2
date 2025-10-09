#include "kseq/kseq.h"
#include "common.h"

// -----------------------------------------------------------------------------
// GPU-safe representations
// -----------------------------------------------------------------------------
struct DeviceKSeq {
        const char* name;
        const char* seq;
        const char* qual;
        int seq_len;
};

struct DeviceMatchResult {
        const char* sample_name;
        const char* signature_name;
        double match_score;
        int integrity_hash;
};

// -----------------------------------------------------------------------------
// GPU kernel
// -----------------------------------------------------------------------------
__global__ void strstr_match_kernel(const DeviceKSeq* samples,
                const DeviceKSeq* signatures,
                DeviceMatchResult* results,
                int num_samples,
                int num_signatures)
{
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        int total = num_samples * num_signatures;
        if (id >= total) return;

        int sample_id = id / num_signatures;
        int sig_id = id % num_signatures;

        const DeviceKSeq& sample = samples[sample_id];
        const DeviceKSeq& signature = signatures[sig_id];

        int best_sum = 0;
        DeviceMatchResult result{};
        result.sample_name = sample.name;
        result.signature_name = signature.name;

        for (int i = 0; i <= sample.seq_len - signature.seq_len; ++i) {
                int curr_sum = 0;

                for (int j = 0; j < signature.seq_len; ++j) {
                        char s = sample.seq[i + j];
                        char t = signature.seq[j];
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
                        result.integrity_hash = best_sum % 97;
                }
        }

        results[id] = result;
}

// -----------------------------------------------------------------------------
// Flatten and upload klibpp::KSeq to GPU (names, seqs, quals)
// -----------------------------------------------------------------------------
void uploadKSeqsToGPU(const std::vector<klibpp::KSeq>& host_kseqs,
                DeviceKSeq*& d_kseqs,
                char*& d_all_names,
                char*& d_all_seqs,
                char*& d_all_quals)
{
        std::string name_concat;
        std::string seq_concat;
        std::string qual_concat;
        std::vector<size_t> name_offsets, seq_offsets;

        size_t name_offset = 0, seq_offset = 0;
        for (const auto& ks : host_kseqs) {
                name_offsets.push_back(name_offset);
                seq_offsets.push_back(seq_offset);

                name_concat += ks.name + '\0'; 
                seq_concat += ks.seq;
                qual_concat += ks.qual.empty() ? std::string(ks.seq.size(), 'I') : ks.qual;

                name_offset += ks.name.size() + 1;
                seq_offset += ks.seq.size();
        }

        // Copy raw data to GPU
        cudaMalloc(&d_all_names, name_concat.size());
        cudaMemcpy(d_all_names, name_concat.data(), name_concat.size(), cudaMemcpyHostToDevice);

        cudaMalloc(&d_all_seqs, seq_concat.size());
        cudaMemcpy(d_all_seqs, seq_concat.data(), seq_concat.size(), cudaMemcpyHostToDevice);

        cudaMalloc(&d_all_quals, qual_concat.size());
        cudaMemcpy(d_all_quals, qual_concat.data(), qual_concat.size(), cudaMemcpyHostToDevice);

        // Build DeviceKSeq array on host
        std::vector<DeviceKSeq> host_dseqs(host_kseqs.size());
        for (size_t i = 0; i < host_kseqs.size(); ++i) {
                host_dseqs[i].name = d_all_names + name_offsets[i];
                host_dseqs[i].seq = d_all_seqs + seq_offsets[i];
                host_dseqs[i].qual = d_all_quals + seq_offsets[i];
                host_dseqs[i].seq_len = host_kseqs[i].seq.size();
        }

        cudaMalloc(&d_kseqs, host_dseqs.size() * sizeof(DeviceKSeq));
        cudaMemcpy(d_kseqs, host_dseqs.data(), host_dseqs.size() * sizeof(DeviceKSeq), cudaMemcpyHostToDevice);
}

// -----------------------------------------------------------------------------
// Main matching 
// -----------------------------------------------------------------------------
void runMatcher(const std::vector<klibpp::KSeq>& samples,
                const std::vector<klibpp::KSeq>& signatures,
                std::vector<MatchResult>& matches)
{
        DeviceKSeq *d_samples = nullptr, *d_signatures = nullptr;
        char *d_all_sample_names = nullptr, *d_all_sample_seqs = nullptr, *d_all_sample_quals = nullptr;
        char *d_all_sig_names = nullptr, *d_all_sig_seqs = nullptr, *d_all_sig_quals = nullptr;

        uploadKSeqsToGPU(samples, d_samples, d_all_sample_names, d_all_sample_seqs, d_all_sample_quals);
        uploadKSeqsToGPU(signatures, d_signatures, d_all_sig_names, d_all_sig_seqs, d_all_sig_quals);

        int num_samples = samples.size();
        int num_signatures = signatures.size();
        int total = num_samples * num_signatures;

        DeviceMatchResult* d_results;
        cudaMalloc(&d_results, total * sizeof(DeviceMatchResult));

        dim3 blockDim(256);
        dim3 gridDim((total + blockDim.x - 1) / blockDim.x);

        strstr_match_kernel<<<gridDim, blockDim>>>(d_samples, d_signatures, d_results, num_samples, num_signatures);
        cudaDeviceSynchronize();

        // Step 1: Copy back raw device results
        std::vector<DeviceMatchResult> tmp(total);
        cudaMemcpy(tmp.data(), d_results, total * sizeof(DeviceMatchResult), cudaMemcpyDeviceToHost);

        // Step 2: Convert to host-friendly MatchResult
        matches.resize(total);
        for (int s = 0; s < num_samples; ++s) {
                for (int t = 0; t < num_signatures; ++t) {
                        int idx = s * num_signatures + t;
                        const auto& dres = tmp[idx];

                        MatchResult hres;
                        hres.sample_name = samples[s].name;
                        hres.signature_name = signatures[t].name;
                        hres.match_score = dres.match_score;
                        hres.integrity_hash = dres.integrity_hash;

                        matches[idx] = std::move(hres);
                }
        }

        // Cleanup GPU allocations
        cudaFree(d_samples);
        cudaFree(d_signatures);
        cudaFree(d_all_sample_names);
        cudaFree(d_all_sample_seqs);
        cudaFree(d_all_sample_quals);
        cudaFree(d_all_sig_names);
        cudaFree(d_all_sig_seqs);
        cudaFree(d_all_sig_quals);
        cudaFree(d_results);
}

