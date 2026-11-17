// ============================================================================
// ds_cnn.h  -  configuration, fixed-point types and weight declarations
// 6-class shape CNN (Assignment 2), structure derived from dbaniya/CNN_guest_lecture
// ============================================================================
#ifndef DS_CNN_H
#define DS_CNN_H

#include "ap_fixed.h"
#include "ap_int.h"

// ---------------- network sizes (must match python/train_cnn.py --arch deep)
static const int IMG_H = 60, IMG_W = 60, N_PIX = IMG_H * IMG_W;   // 3600
static const int N_CLASSES = 6;

static const int C1 = 16;                  // conv1 channels
static const int H1 = 30, W1 = 30;         // after conv1 + maxpool
static const int C2 = 32;                  // block 2 (dw 16 -> pw 32) + maxpool
static const int H2 = 15, W2 = 15;
static const int C3 = 32;                  // block 3 (dw 32 -> pw 32) + maxpool (15 -> 7, floor)
static const int H3 = 7, W3 = 7;
static const int C4 = 32;                  // block 4 (dw 32 -> pw 32), then global average

// MACs per clock cycle inside each depthwise-separable block (DSP budget:
// Zynq-7020 has 220 DSP48E1). Must divide the input channel count.
static const int PAR2 = 16;                // 900 pixels -> fully parallel over 16 channels
static const int PAR3 = 8;                 // 225 pixels
static const int PAR4 = 8;                 //  49 pixels

// ---------------- fixed-point types ----------------------------------------
// data_t : the IP's input pixels and output logits, Q8.8 (int16 = value*256),
//          which is exactly what the ARM writes / reads.
// feat_t : internal feature maps. Same 16 bits; rounding + saturation instead
//          of the default truncation + wrap-around.
// wgt_t  : BN-folded weights / biases, 18 bits = native DSP48E1 multiplier
//          width, range +-64, resolution 2^-11.
// acc_t  : MAC accumulators, 20 fractional bits = exact for data_t x wgt_t
//          products. No saturation logic here: it would sit on the
//          loop-carried add and limit Fmax (|sums| stay far below 2048).
typedef ap_fixed<16, 8>                  data_t;
typedef ap_fixed<16, 8, AP_RND, AP_SAT>  feat_t;
typedef ap_fixed<18, 7, AP_RND, AP_SAT>  wgt_t;
typedef ap_fixed<32, 12>                 acc_t;

// ---------------- top-level function ---------------------------------------
// input_img : 3600 Q8.8 pixels in DDR, read through m_axi (bundle gmem)
// pred_out  : predicted class index           (AXI-Lite register)
// logits_out: 6 class scores, Q8.8            (AXI-Lite registers)
void ds_cnn(const data_t *input_img, int *pred_out, data_t logits_out[N_CLASSES]);

// ---------------- trained parameters (ds_cnn_weights_data.h) ---------------
extern const wgt_t CONV1_W[9][C1];
extern const wgt_t CONV1_B[C1];
extern const wgt_t DW2_W[C1][9];
extern const wgt_t DW2_B[C1];
extern const wgt_t PW2_W[C1][C2];
extern const wgt_t PW2_B[C2];
extern const wgt_t DW3_W[C2][9];
extern const wgt_t DW3_B[C2];
extern const wgt_t PW3_W[C2][C3];
extern const wgt_t PW3_B[C3];
extern const wgt_t DW4_W[C3][9];
extern const wgt_t DW4_B[C3];
extern const wgt_t PW4_W[C3][C4];
extern const wgt_t PW4_B[C4];
extern const wgt_t FC_W[C4][N_CLASSES];
extern const wgt_t FC_B[N_CLASSES];

#endif  // DS_CNN_H
