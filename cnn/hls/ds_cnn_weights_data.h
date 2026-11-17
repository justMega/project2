// ============================================================================
// ds_cnn_weights_data.h  -  trained, BN-folded parameters
// The .txt files are written by python/train_cnn.py into
// <run>/weights_cstyle_folded/ ; copy that folder next to this header.
// Include this file in exactly ONE translation unit (ds_cnn.cpp).
// ============================================================================
#ifndef DS_CNN_WEIGHTS_DATA_H
#define DS_CNN_WEIGHTS_DATA_H

#include "ds_cnn.h"

const wgt_t CONV1_W[9][C1] =
#include "weights_cstyle_folded/conv1_W.txt"
;
const wgt_t CONV1_B[C1] =
#include "weights_cstyle_folded/conv1_B.txt"
;
const wgt_t DW2_W[C1][9] =
#include "weights_cstyle_folded/dw2_W.txt"
;
const wgt_t DW2_B[C1] =
#include "weights_cstyle_folded/dw2_B.txt"
;
const wgt_t PW2_W[C1][C2] =
#include "weights_cstyle_folded/pw2_W.txt"
;
const wgt_t PW2_B[C2] =
#include "weights_cstyle_folded/pw2_B.txt"
;
const wgt_t DW3_W[C2][9] =
#include "weights_cstyle_folded/dw3_W.txt"
;
const wgt_t DW3_B[C2] =
#include "weights_cstyle_folded/dw3_B.txt"
;
const wgt_t PW3_W[C2][C3] =
#include "weights_cstyle_folded/pw3_W.txt"
;
const wgt_t PW3_B[C3] =
#include "weights_cstyle_folded/pw3_B.txt"
;
const wgt_t DW4_W[C3][9] =
#include "weights_cstyle_folded/dw4_W.txt"
;
const wgt_t DW4_B[C3] =
#include "weights_cstyle_folded/dw4_B.txt"
;
const wgt_t PW4_W[C3][C4] =
#include "weights_cstyle_folded/pw4_W.txt"
;
const wgt_t PW4_B[C4] =
#include "weights_cstyle_folded/pw4_B.txt"
;
const wgt_t FC_W[C4][N_CLASSES] =
#include "weights_cstyle_folded/fc_W.txt"
;
const wgt_t FC_B[N_CLASSES] =
#include "weights_cstyle_folded/fc_B.txt"
;

#endif  // DS_CNN_WEIGHTS_DATA_H
