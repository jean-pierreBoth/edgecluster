## Results

Computations are done on 32 threads cpu.

### Mnist Digits

As we rescale mnist image to one pixel (division by 28*28) the cost given here have been reset by
multiplying the code results by 784 to compare to kmedoids result.
Note that  kmedoids crate have images pixel divided by 256
so they have been also reset by 256 to do comparisons. (See file [kmedoids](https://github.com/kno10/rust-kmedoids/blob/main/examples/mnist-kmedoids.rs))

### Without dimension reduction


The algorithm implemented need 2s system time and 16s cpu to compute the 3 partitons.
We compare with serial faster_pam which needs 790s distance computation and 80s for pam iterations for the partition 
of size 10.


| partition size | cost l2 (medoid) |   faster_pam   |  nmi  |
| :------------: | :--------------: | :------------: | :---: |
|       10       |                  | 1.123 $10^{8}$ |       |
|       15       |                  |                |       |
|       25       |                  |                |       |

### With dimension reduction to 3

### Song benchmark

### Higgs data

We used 3 clustering sizes : 10, 100 and 200.
For each size we compute Nmi (sqrt) medoid l2 cost, and finally l2 cost to mean of each cluster

TODO: rappeler les definitions (sqrt/pas sqrt)

| partition size | cost l2 (medoid) | cost l2 (kmean) |      nmi      |
| :------------: | :--------------: | :-------------: | :-----------: |
|       10       |    5.16  10^7    |  1.78 $10^{8}$  | 6.0 $10^{-4}$ |
|      100       |    4.64  10^7    |  1.39 $10^{8}$  | 5.0 $10^{-3}$ |
|      200       |   4.52  $10^7$   |  1.31 $10^{8}$  | 6.0 $10^{-3}$ |

Times needed to collect the 3 partitions:
- cpu(s) : 2943
- sys(s) : 267
  
Using the *kmean* provided by the Julia package [clustering](https://juliastats.org/Clustering.jl/stable/algorithms.html) we get a cost (inertia) of 1.53 10^8 in  cpu time 304s and sys time(s) 64 for the partition of size 200.


| partition size |  kmean inertia  | cpu(s), sys(s) |
| :------------: | :-------------: | :------------: |
|       10       |  1.53 $10^{8}$  |    304, 68     |
|      100       |  1.11 $10^{8}$  |   2280, 770    | unconverged |
|      200       | 0.998  $10^{8}$ |   4400, 1540   |


Our kmean cost is within 15% for the partiton size 10, and within 25, 30% for the partitions size of 100 ans 200 but these are obtained at no cost. The nmi are not good but Higgs data are not clusterizable
