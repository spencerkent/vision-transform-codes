vision-transform-codes
======================
A set of implementations of different image transform codes (sparse coding, ICA, PCA, etc.) to facilitate comparison between them.
The implementation is in the PyTorch GPGPU framework and should be relatively performant in terms of wall-clock time for training
and inference. The modular organization is to stress the interchangibility of different techniques for code inference (analysis)
and reconstruction (synthesis). 

This also includes some basic image processing utilities that may be useful to someone working with images in Python, particularly when fitting transform codes to image datasets. 

Of particular interest is code for doing **convolutional** sparse coding and for enforcing "subspace" or "topographic" constraints on sparse codes. The convolutional variants of sparse coding are implemented in the `convolutional/` sudirectories of `analysis_transforms` and `dict_update_rules`. The subspace analysis transform is implemented in `analysis_transforms/fully_connected/subspace_ista_fista.py`. The topographic sparse coding implementation lives on the `topographic-sparse-coding` branch, I haven't yet merged it into `master`.

### Authors
Spencer Kent
