#include "cv_kcf.h"

CVKCF::CVKCF()
{
    detect_thresh = 0.5f;
    sigma=0.2f;
    lambda=0.0001f;
    interp_factor=0.075f;
    output_sigma_factor=1.0f / 16.0f;
    resize=true;
    max_patch_size=80*80;
    split_coeff=true;
    wrap_kernel=false;
    desc_npca = 1;
    desc_pca = 2;

    //feature compression
    compress_feature=true;
    compressed_size=2;
    pca_learning_rate=0.15f;
}