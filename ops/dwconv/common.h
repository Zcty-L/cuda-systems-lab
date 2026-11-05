#pragma once

struct Conv2DParam
{
    int in_h;
    int in_w;
    int in_ch;
    int inHW;
    int inChKhKw;
    int inBatchNumel;
    int out_ch;
    int out_h;
    int out_w;
    int outHW;
    int outBatchNumel;
    int Kh;
    int Kw;
    int KhKw;
    int Sh;
    int Sw;
    int Ph;
    int Pw;

    int k_tiles;
    int first_k_tile;

    int kernelWeightsCount;
    int biasWeightsCount;

    bool use_bias = true;
};
