#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

// GPU-safe version
struct DeviceKSeq {
        const char* name;
        const char* seq;
        const char* qual;
        int seq_len;
};

// Upload simple KSeq list to GPU
void uploadKSeqsToGPU(const std::vector<KSeq>& host_kseqs,
                DeviceKSeq*& d_kseqs)
{
        std::vector<DeviceKSeq> host_dseqs;

        for (const auto& ks : host_kseqs) {
                DeviceKSeq dks{};
                dks.seq_len = ks.seq.size();

                // Allocate device memory for each field
                char* d_name; cudaMalloc(&d_name, ks.name.size() + 1);
                cudaMemcpy(d_name, ks.name.c_str(), ks.name.size() + 1, cudaMemcpyHostToDevice);

                char* d_seq; cudaMalloc(&d_seq, ks.seq.size());
                cudaMemcpy(d_seq, ks.seq.data(), ks.seq.size(), cudaMemcpyHostToDevice);

                char* d_qual; cudaMalloc(&d_qual, ks.qual.size());
                cudaMemcpy(d_qual, ks.qual.data(), ks.qual.size(), cudaMemcpyHostToDevice);

                dks.name = d_name;
                dks.seq = d_seq;
                dks.qual = d_qual;
                host_dseqs.push_back(dks);
        }

        // Copy DeviceKSeq array to GPU
        cudaMalloc(&d_kseqs, host_dseqs.size() * sizeof(DeviceKSeq));
        cudaMemcpy(d_kseqs, host_dseqs.data(),
                        host_dseqs.size() * sizeof(DeviceKSeq),
                        cudaMemcpyHostToDevice);
}

int main() {
        std::vector<KSeq> samples = {
                {"sample1", "ACGT", "IIII"},
                {"sample2", "GGAA", "JJJJ"}
        };

        DeviceKSeq* d_samples = nullptr;
        uploadKSeqsToGPU(samples, d_samples);

        std::cout << "Uploaded " << samples.size() << " KSeqs to GPU.\n";

        cudaFree(d_samples);
        return 0;
}

