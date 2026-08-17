# Research status

This repository contains a completed pilot investigation of context sparsity,
model order, and estimator choice in finite-data n-gram language models.

The initial research question was narrowed after literature review showed that
the broad interaction among n-gram order, training-data size, smoothing, and
generalisation is well established.

The pilot produced two main observations:

1. singleton-history occupancy closely tracked held-out unseen-history rates;
2. generalisation reversals observed under fixed-order additive smoothing
   disappeared under validation-selected lower-order interpolation.

The pilot additionally exposed a confound between training-set size and
occupancy statistics, motivating a subsequent study designed to isolate
training-only predictors of marginal context value at fixed data size and
model order.