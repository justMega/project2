// ============================================================================
// ds_cnn_tb.cpp  -  C-simulation / co-simulation testbench
//
// Reads the test images exported by python/train_cnn.py (tb_data/):
//   images_u8.bin        N x 60 x 60 uint8 (row-major)
//   labels_i32.bin       N int32 true labels
//   keras_logits_f32.bin N x 6 float32 logits of the trained Keras model
// Converts every image to Q8.8 exactly like the bare-metal ARM application,
// runs ds_cnn(), and compares
//   (a) the HLS prediction with the true label   -> accuracy
//   (b) the HLS prediction with the Keras one    -> agreement (bit-accuracy check)
//   (c) the HLS logits with the Keras logits     -> max / mean abs error
// Writes hls_results.csv. Returns 0 (PASS) if agreement >= 98 %.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "ds_cnn.h"

static const char *CLASS_NAMES[N_CLASSES] = {"circle", "square", "fig_eight",
                                             "rectangle", "triangle", "elipse"};

template <typename T>
static bool read_bin(const std::string &path, std::vector<T> &v) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize(n / sizeof(T));
    size_t got = fread(v.data(), sizeof(T), v.size(), f);
    fclose(f);
    return got == v.size();
}

// The csim working directory differs between tool versions, so look upwards.
static std::string find_tb_dir(int argc, char **argv) {
    std::vector<std::string> cands;
    if (argc > 1) cands.push_back(argv[1]);
#ifdef TB_DATA_DIR
    cands.push_back(TB_DATA_DIR);
#endif
    std::string up = "";
    for (int i = 0; i < 8; i++) {
        cands.push_back(up + "tb_data/");
        up += "../";
    }
    for (auto &d : cands) {
        std::string p = d;
        if (!p.empty() && p.back() != '/') p += '/';
        FILE *f = fopen((p + "images_u8.bin").c_str(), "rb");
        if (f) { fclose(f); return p; }
    }
    return "";
}

int main(int argc, char **argv) {
    std::string dir = find_tb_dir(argc, argv);
    if (dir.empty()) {
        printf("ERROR: tb_data/ not found (pass its path as argument or -DTB_DATA_DIR)\n");
        return 2;
    }
    std::vector<uint8_t> imgs;
    std::vector<int32_t> labels;
    std::vector<float> klog;
    if (!read_bin(dir + "images_u8.bin", imgs) || !read_bin(dir + "labels_i32.bin", labels) ||
        !read_bin(dir + "keras_logits_f32.bin", klog)) {
        printf("ERROR: cannot read files in %s\n", dir.c_str());
        return 2;
    }
    const int n = (int)labels.size();
    if ((int)imgs.size() != n * N_PIX || (int)klog.size() != n * N_CLASSES) {
        printf("ERROR: inconsistent tb_data sizes\n");
        return 2;
    }
    printf("Testbench: %d images from %s\n", n, dir.c_str());

    static data_t in[N_PIX];
    int correct = 0, agree = 0;
    double max_err = 0, sum_err = 0;
    int cm[N_CLASSES][N_CLASSES] = {{0}};
    FILE *csv = fopen("hls_results.csv", "w");
    if (csv) {
        fprintf(csv, "idx,true,hls_pred,keras_pred");
        for (int c = 0; c < N_CLASSES; c++) fprintf(csv, ",hls_logit_%d", c);
        fprintf(csv, "\n");
    }

    for (int k = 0; k < n; k++) {
        // u8 -> Q8.8, identical integer formula to the ARM application:
        //   q = round(1024*x/255) - 512   (value = q / 256 = 4x/255 - 2)
        for (int i = 0; i < N_PIX; i++) {
            int px = imgs[(size_t)k * N_PIX + i];
            int16_t q = (int16_t)((1024 * px + 127) / 255 - 512);
            in[i].range(15, 0) = (ap_uint<16>)(uint16_t)q;
        }
        int pred = -1;
        data_t logits[N_CLASSES];
        ds_cnn(in, &pred, logits);

        const float *kl = &klog[(size_t)k * N_CLASSES];
        int kpred = 0;
        for (int c = 1; c < N_CLASSES; c++)
            if (kl[c] > kl[kpred]) kpred = c;
        for (int c = 0; c < N_CLASSES; c++) {
            double e = std::fabs(logits[c].to_double() - kl[c]);
            sum_err += e;
            if (e > max_err) max_err = e;
        }
        correct += (pred == labels[k]);
        agree += (pred == kpred);
        if (pred >= 0 && pred < N_CLASSES) cm[labels[k]][pred]++;

        if (csv) {
            fprintf(csv, "%d,%d,%d,%d", k, labels[k], pred, kpred);
            for (int c = 0; c < N_CLASSES; c++) fprintf(csv, ",%.5f", logits[c].to_double());
            fprintf(csv, "\n");
        }
        if (pred != kpred)
            printf("  [diff] img %d: true=%s hls=%s keras=%s\n", k, CLASS_NAMES[labels[k]],
                   CLASS_NAMES[pred], CLASS_NAMES[kpred]);
    }
    if (csv) fclose(csv);

    printf("\nConfusion matrix (rows = true, cols = HLS prediction)\n%-10s", "");
    for (int c = 0; c < N_CLASSES; c++) printf(" %9.9s", CLASS_NAMES[c]);
    printf("\n");
    for (int r = 0; r < N_CLASSES; r++) {
        printf("%-10s", CLASS_NAMES[r]);
        for (int c = 0; c < N_CLASSES; c++) printf(" %9d", cm[r][c]);
        printf("\n");
    }
    double agr = 100.0 * agree / n;
    printf("\nHLS accuracy vs labels     : %.2f %% (%d/%d)\n", 100.0 * correct / n, correct, n);
    printf("HLS vs Keras agreement     : %.2f %% (%d/%d)\n", agr, agree, n);
    printf("Logit error vs Keras       : max %.4f, mean %.4f\n", max_err,
           sum_err / (n * N_CLASSES));
    printf("%s\n", agr >= 98.0 ? "*** TEST PASSED ***" : "*** TEST FAILED ***");
    return agr >= 98.0 ? 0 : 1;
}
