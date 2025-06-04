## Results

### Mnist Digits

### Higgs data

We used 3 clustering sizes : 10, 100 and 200.
For each size we compute Nmi (sqrt) medoid l2 cost, and finally l2 cost to mean of each cluster


| partition size | cost l2 (medoid) | cost l2 (kmean) |      nmi      |
| :------------: | :--------------: | :-------------: | :-----------: |
|       10       |    5.16  10^7    |   4.325 10^7    | 6.0 $10^{-4}$ |
|      100       |    4.64  10^7    |   3.823 10^7    | 5.0 $10^{-3}  |
|      200       |   4.52  $10^7$   |   3.728 10^7    | 6.0 $10^{-3}$ |

Times needed to collect the 3 partitions:
- cpu(s) : 2943
- sys(s) : 267
  
Using the *kmean* provided by the crate [clustering](https://crates.io/crates/clustering) we get a cost of 3.24 10^7 in  cpu time 3.74e4 and sys time(s) 2.31e4 for a partition of size 200.

Nmi shows that the clusterization do not bring much segregation of data but our kmean cost is within 15% of the standard kmean at a negligible cost. 