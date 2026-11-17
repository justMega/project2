// ============================================================================
// ds_cnn.cpp  -  Vitis HLS accelerator for the 6-class shape CNN
//
// Network (python/train_cnn.py --arch deep, BN folded into the weights):
//   Conv3x3 1->16  + ReLU + MaxPool2   60x60x1  -> 30x30x16   (f1)
//   DS block 2: DW3x3 16, PW1x1 16->32, ReLU, MaxPool2 -> 15x15x32 (f2)
//   DS block 3: DW3x3 32, PW1x1 32->32, ReLU, MaxPool2 ->  7x7x32  (f3)
//   DS block 4: DW3x3 32, PW1x1 32->32, ReLU, global average   -> 32
//   Dense 32->6 -> logits -> argmax
//
// Design choices
//   * All feature maps stay on chip (BRAM/LUTRAM), partitioned along the
//     channel dimension so that PAR channels are read in the same cycle.
//   * Each DS block processes one pixel at a time: depthwise (PAR MACs/cycle)
//     then pointwise (PAR MACs/cycle); ReLU and MaxPool are fused into the
//     write-back, so the un-pooled map is never stored.
//   * The last block accumulates the global average directly (no 7x7 map).
//   * Rounding + saturating fixed point (see ds_cnn.h).
// ============================================================================
#include "ds_cnn.h"
#include "ds_cnn_weights_data.h"

typedef ap_fixed<24, 14> sum_t;                 // global-average accumulator (49 x < 128)

static inline feat_t relu(acc_t x) {
#pragma HLS INLINE
    return (x > 0) ? (feat_t)x : (feat_t)0;
}

// ----------------------------------------------------------------------------
// Conv3x3 (same padding, 1 input channel) + ReLU + MaxPool2x2
// For each pooled output the 4x4 input patch is loaded into registers once;
// then one output channel per clock cycle (2x2 positions x 9 taps = 36 MACs).
// ----------------------------------------------------------------------------
static void conv1_relu_maxpool(const data_t img[IMG_H][IMG_W], feat_t out[H1][W1][C1]) {
#pragma HLS INLINE off
    data_t patch[4][4];
#pragma HLS ARRAY_PARTITION variable=patch complete dim=0

C1_ROW:
    for (int ph = 0; ph < H1; ph++) {
    C1_COL:
        for (int pw = 0; pw < W1; pw++) {
        C1_PATCH:
            for (int i = 0; i < 16; i++) {
#pragma HLS PIPELINE II=1
                int r = 2 * ph - 1 + (i >> 2);
                int c = 2 * pw - 1 + (i & 3);
                bool inside = (r >= 0) && (r < IMG_H) && (c >= 0) && (c < IMG_W);
                patch[i >> 2][i & 3] = inside ? img[inside ? r : 0][inside ? c : 0] : (data_t)0;
            }
        C1_OC:
            for (int oc = 0; oc < C1; oc++) {
#pragma HLS PIPELINE II=1
                feat_t maxv = 0;                 // ReLU outputs are >= 0
                for (int p = 0; p < 4; p++) {
                    int dy = p >> 1, dx = p & 1;
                    acc_t acc = CONV1_B[oc];
                    for (int kh = 0; kh < 3; kh++)
                        for (int kw = 0; kw < 3; kw++)
                            acc += patch[dy + kh][dx + kw] * CONV1_W[kh * 3 + kw][oc];
                    feat_t v = relu(acc);
                    if (v > maxv) maxv = v;
                }
                out[ph][pw][oc] = maxv;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// One output pixel of a depthwise-separable block:
//   depthwise 3x3 (same padding) + ReLU, pointwise 1x1 + ReLU  ->  pw_out[COUT]
// Channels are processed in NB groups of PAR. Inside the pipelined loops each
// unrolled lane j owns a fixed accumulator, so no wide multiplexer sits on the
// loop-carried path (timing at 100 MHz).
// ----------------------------------------------------------------------------
template <int H, int W, int CIN, int COUT, int PAR>
static void ds_pixel(const feat_t in[H][W][CIN], int h, int w,
                     const wgt_t dwW[CIN][9], const wgt_t dwB[CIN],
                     const wgt_t pwW[CIN][COUT], const wgt_t pwB[COUT],
                     feat_t pw_out[COUT]) {
#pragma HLS INLINE
    const int NB = CIN / PAR;                    // channel groups
    acc_t dacc[PAR];
#pragma HLS ARRAY_PARTITION variable=dacc complete
    feat_t dw_out[CIN];
#pragma HLS ARRAY_PARTITION variable=dw_out complete

DW_G:
    for (int b = 0; b < NB; b++) {
    DW_T:
        for (int t = 0; t < 9; t++) {
#pragma HLS PIPELINE II=1
            int ih = h + t / 3 - 1, iw = w + t % 3 - 1;
            bool inside = (ih >= 0) && (ih < H) && (iw >= 0) && (iw < W);
            for (int j = 0; j < PAR; j++) {
                int ic = b * PAR + j;
                feat_t x = inside ? in[inside ? ih : 0][inside ? iw : 0][ic] : (feat_t)0;
                acc_t prev = (t == 0) ? (acc_t)dwB[ic] : dacc[j];
                acc_t s = prev + x * dwW[ic][t];
                dacc[j] = s;
                if (t == 8) dw_out[ic] = relu(s);
            }
        }
    }

    acc_t acc = 0;
PW_OC:
    for (int oc = 0; oc < COUT; oc++) {
    PW_G:
        for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II=1
            acc_t s = 0;                         // adder tree over the PAR lanes
            for (int j = 0; j < PAR; j++)
                s += dw_out[b * PAR + j] * pwW[b * PAR + j][oc];
            acc_t a = ((b == 0) ? (acc_t)pwB[oc] : acc) + s;
            acc = a;
            if (b == NB - 1) pw_out[oc] = relu(a);
        }
    }
}

// DS block followed by MaxPool 2x2 (floor, like Keras 'valid' pooling)
template <int H, int W, int CIN, int COUT, int PAR>
static void ds_block_pool(const feat_t in[H][W][CIN], feat_t out[H / 2][W / 2][COUT],
                          const wgt_t dwW[CIN][9], const wgt_t dwB[CIN],
                          const wgt_t pwW[CIN][COUT], const wgt_t pwB[COUT]) {
#pragma HLS INLINE off
    const int HO = H / 2, WO = W / 2;
INIT:
    for (int i = 0; i < HO * WO; i++) {
#pragma HLS PIPELINE II=1
        for (int oc = 0; oc < COUT; oc++) out[i / WO][i % WO][oc] = 0;   // ReLU >= 0
    }
ROW:
    for (int h = 0; h < 2 * HO; h++) {           // last odd row/col is dropped by the pool
    COL:
        for (int w = 0; w < 2 * WO; w++) {
            feat_t pw_out[COUT];
#pragma HLS ARRAY_PARTITION variable=pw_out complete
            ds_pixel<H, W, CIN, COUT, PAR>(in, h, w, dwW, dwB, pwW, pwB, pw_out);
            for (int oc = 0; oc < COUT; oc++) {
#pragma HLS UNROLL
                feat_t o = out[h >> 1][w >> 1][oc];
                out[h >> 1][w >> 1][oc] = (pw_out[oc] > o) ? pw_out[oc] : o;
            }
        }
    }
}

// DS block followed by global average pooling
template <int H, int W, int CIN, int COUT, int PAR>
static void ds_block_gap(const feat_t in[H][W][CIN], feat_t gap[COUT],
                         const wgt_t dwW[CIN][9], const wgt_t dwB[CIN],
                         const wgt_t pwW[CIN][COUT], const wgt_t pwB[COUT]) {
#pragma HLS INLINE off
    sum_t gsum[COUT];
#pragma HLS ARRAY_PARTITION variable=gsum complete
    for (int oc = 0; oc < COUT; oc++) {
#pragma HLS UNROLL
        gsum[oc] = 0;
    }
ROW:
    for (int h = 0; h < H; h++) {
    COL:
        for (int w = 0; w < W; w++) {
            feat_t pw_out[COUT];
#pragma HLS ARRAY_PARTITION variable=pw_out complete
            ds_pixel<H, W, CIN, COUT, PAR>(in, h, w, dwW, dwB, pwW, pwB, pw_out);
            for (int oc = 0; oc < COUT; oc++) {
#pragma HLS UNROLL
                gsum[oc] += pw_out[oc];
            }
        }
    }
    const ap_ufixed<18, 0> inv_n = 1.0 / (H * W);
SCALE:
    for (int oc = 0; oc < COUT; oc++) {          // one shared multiplier
#pragma HLS PIPELINE II=1
        gap[oc] = (feat_t)(gsum[oc] * inv_n);
    }
}

// ----------------------------------------------------------------------------
// Dense 32 -> 6 + argmax (sequential: negligible latency, minimal DSP)
// ----------------------------------------------------------------------------
static int classify(const feat_t gap[C4], data_t logits_out[N_CLASSES]) {
#pragma HLS INLINE off
    int best = 0;
    acc_t best_v = 0;
FC_O:
    for (int o = 0; o < N_CLASSES; o++) {
        acc_t acc = FC_B[o];
    FC_I:
        for (int ic = 0; ic < C4; ic++) {        // 192 cycles in total, 1 DSP
#pragma HLS PIPELINE II=1
            acc += gap[ic] * FC_W[ic][o];
        }
        logits_out[o] = (data_t)(feat_t)acc;     // saturate to Q8.8
        if (o == 0 || acc > best_v) {            // first maximum wins (numpy.argmax)
            best_v = acc;
            best = o;
        }
    }
    return best;
}

// ----------------------------------------------------------------------------
// Top level
// ----------------------------------------------------------------------------
void ds_cnn(const data_t *input_img, int *pred_out, data_t logits_out[N_CLASSES]) {
#pragma HLS INTERFACE m_axi     port=input_img offset=slave bundle=gmem depth=3600 max_read_burst_length=256
#pragma HLS INTERFACE s_axilite port=input_img  bundle=control
#pragma HLS INTERFACE s_axilite port=pred_out   bundle=control
#pragma HLS INTERFACE s_axilite port=logits_out bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    // ---- on-chip feature memories
    static data_t img[IMG_H][IMG_W];
    static feat_t f1[H1][W1][C1];
    static feat_t f2[H2][W2][C2];
    static feat_t f3[H3][W3][C3];
#pragma HLS BIND_STORAGE variable=img type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=f1 complete dim=3
#pragma HLS ARRAY_PARTITION variable=f2 complete dim=3
#pragma HLS ARRAY_PARTITION variable=f3 complete dim=3

    // ---- weight ROMs: split along the dimension read in parallel
#pragma HLS ARRAY_PARTITION variable=CONV1_W complete dim=1
#pragma HLS ARRAY_PARTITION variable=DW2_W complete dim=1
#pragma HLS ARRAY_PARTITION variable=DW3_W complete dim=1
#pragma HLS ARRAY_PARTITION variable=DW4_W complete dim=1
#pragma HLS ARRAY_PARTITION variable=DW2_B complete
#pragma HLS ARRAY_PARTITION variable=DW3_B complete
#pragma HLS ARRAY_PARTITION variable=DW4_B complete
#pragma HLS ARRAY_PARTITION variable=PW2_W complete dim=1
#pragma HLS ARRAY_PARTITION variable=PW3_W complete dim=1
#pragma HLS ARRAY_PARTITION variable=PW4_W complete dim=1

    feat_t gap[C4];

    // 1) burst-read the Q8.8 image from DDR into BRAM
READ_IMG:
    for (int i = 0; i < N_PIX; i++) {
#pragma HLS PIPELINE II=1
        img[i / IMG_W][i % IMG_W] = input_img[i];
    }

    // 2) feature extraction
    conv1_relu_maxpool(img, f1);
    ds_block_pool<H1, W1, C1, C2, PAR2>(f1, f2, DW2_W, DW2_B, PW2_W, PW2_B);
    ds_block_pool<H2, W2, C2, C3, PAR3>(f2, f3, DW3_W, DW3_B, PW3_W, PW3_B);
    ds_block_gap<H3, W3, C3, C4, PAR4>(f3, gap, DW4_W, DW4_B, PW4_W, PW4_B);

    // 3) classification
    *pred_out = classify(gap, logits_out);
}
