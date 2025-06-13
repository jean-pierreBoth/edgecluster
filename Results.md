## Results

Computations are done on a 32 threads cpu.

### Mnist Digits

As we rescale mnist image to one pixel (division by 28*28) the cost given here have been reset by
multiplying the code results by 784 to compare to kmedoids result.  
Note that  kmedoids crate have images pixel divided by 256
so they have been also reset by 256 to do comparisons. (See file [kmedoids](https://github.com/kno10/rust-kmedoids/blob/main/examples/mnist-kmedoids.rs))
Moreover the cost are normalized by the size of the sample, 60000 data corresponding to file *train-images-idx3-ubyte* for faster_pam and 70000 for edgecluster.

### par_fasterpam

| partition size | faster_pam cost, sys time(s) |
| :------------: | :--------------------------: |
|       10       |           1873, 30           |
|       15       |           1807, 63           |
|       25       |          1725, 1725          |


**faster_pam times do not take into account the time needed to computes distances**

### Without dimension reduction


The algorithm implemented need 2s system time and 16s cpu to **compute the 3 partitons**.
Results are averaged on 5 runs.


| partition size | cost l2 (medoid) |  nmi  |
| :------------: | :--------------: | :---: |
|       10       |       2070       | 0.26  |
|       15       |       2010       | 0.30  |
|       25       |       1930       | 0.34  |

Our cost is 11% higher than par_fasterpam, the system times are orders of magnitude lower.

### With dimension reduction to 3

We test the impact of reducing dimension to 3.

The algorithm implemented need 1s system time and 16s cpu to **compute the 3 partitons**.
Results are averaged on 5 runs.

| partition size | cost l2 (medoid) |  nmi  |
| :------------: | :--------------: | :---: |
|       10       |       2067       | 0.25  |
|       15       |       2016       | 0.30  |
|       25       |       1921       | 0.35  |

The dimension do not affect the result, but still reduces times.
### Song benchmark


Data can be found at [UCI](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd).
It consists in 515345 vectors of dimension 90.

| partition size | cost l2 (medoid) |
| :------------: | :--------------: |
|       10       |     9.49 10⁸     |
|      100       |     8.18 10⁸     |
|      1000      |     6.66 10⁸     |

sys time 140s, cpu time 2900s.

The results are slightly better than those in the original paper as we do one supplementary pass at the end of the algoritmm.
We keep all medoids found and redispatch other points to nearest medoid.

### Higgs data

We used 3 clustering sizes : 10, 100 and 200.
For each size we compute Nmi (sqrt) as provided by  crate[coreset](https://crates.io/crates/coreset) medoid l2 cost.
We also compute a l2 cost (inertia) to barycenter of each cluster as in kmean to see how far a cost computed from medoid affectation can be of standard result.

TODO: rappeler les definitions (sqrt/pas sqrt)

| partition size | cost l2 (medoid) | cost l2 (kmean) |      nmi      |
| :------------: | :--------------: | :-------------: | :-----------: |
|       10       |   5.16  $10^7$   |  1.78 $10^{8}$  | 6.0 $10^{-4}$ |
|      100       |   4.64  $10^7$   |  1.39 $10^{8}$  | 5.0 $10^{-3}$ |
|      200       |   4.52  $10^7$   |  1.31 $10^{8}$  | 6.0 $10^{-3}$ |

Times needed to collect the 3 partitions:
- cpu(s) : 2943
- sys(s) : 267
  
Using the *kmean* provided by the Julia package [clustering](https://juliastats.org/Clustering.jl/stable/algorithms.html) we get a cost (inertia) of 1.53 10^8 in  cpu time of 304s and sys time of 64s for the partition of size 200.


| partition size |  kmean inertia  | cpu(s), sys(s) |
| :------------: | :-------------: | :------------: |
|       10       |  1.53 $10^{8}$  |    304, 68     |
|      100       |  1.11 $10^{8}$  |   2280, 770    | unconverged |
|      200       | 0.998  $10^{8}$ |   4400, 1540   |


Our kmean cost is within 15% for the partiton size 10, and within 25, 30% for the partitions size of 100 ans 200 but these are obtained at no cost. The nmi are not good but Higgs data are not clusterizable

#### Another remark

The reason why it is impossible to reach one point by cell even with 31 layers is that Higgs data have 270520 cells with 2 identical points.
**The duplicated points have all label 0**. This can be checked by running example *higgs.rs* with RUST_LOG=debug. (You will get
a large dump!!) The are about 4500 cells with more than one point but where all points are not identical.