# h-clustering

This crate is dedicated to recent k-median (hierarchical) clustering. It is related to the crate [coreset](https://crates.io/crates/coreset)
as it provides efficient approximate clustering but in a different context. It also uses sub-crates of *coreset* such as the implementation
of Normalized Information to assess cluster merit.

For now we implement:

1) Parallel and Efficient Hierarchical k-Median Clustering 
  V Cohen-Addad S. Lattanzi et al. 2021 [CAL](https://dl.acm.org/doi/10.5555/3540261.3541816)


## Introduction

The algorithm first dispatch points in **2-hierarchically separated tree** and then builds a clustering
by analyzing the cost/benefit of merging cells at various level of the tree.
This has 2 consequences:
    - **The embedding can be used to build partitions of data of different sizes  at marginal cost** (See results for Higgs data).
    - In case of very high dimensional data the division of cells edges by 2 in each dimension can be costly.  
      In this case we can reduce data dimension with the module *smalld* based on [skorski](https://proceedings.mlr.press/v134/skorski21a/skorski21a.pdf) and [mezzadri](https://arxiv.org/pdf/math-ph/0609050).

The optimal cost is approximated within
$$ 
O(min(d, log k) \  log ∆)
$$ 
with : 
  - d : space dimension
  - k : number of cluster asked in a partiton
  - ∆ : max length of box edge enclosing data.
  
The algorithm as the one implemented in crate [coreset](https://crates.io/crates/coreset) provides a very decent approximation of the optimal cost, but at a speed at least one order of magnitude faster than standard kmean or kmedoid
as can be seen in [Higgs](https://archive.ics.uci.edu/dataset/280/higgs) data.

## Results

Results are presented for Mnist data and Higgs data in 
## License

Licensed under either of

* Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
* MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.
