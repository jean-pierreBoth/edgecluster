## Results

### Mnist Digits

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
  
Using the *kmean* provided by the Julia package [clustering](https://juliastats.org/Clustering.jl/stable/algorithms.html) we get a cost (inertia) of 1.53 10^8 in  cpu time 304s and sys time(s) 64 for a partition of size 200.


| partition size |  kmean inertia   | cpu(s), sys(s) |
| :------------: | :--------------: | :-----------:  | 
|       10       |  1.53 $10^{8}$   |    304,68      |
|      100       |  1.11 $10^{8}$   |   2280, 770    |  unconverged
|      200       |                  |                | 


Nmi shows that the clusterization do not bring much segregation of data but our kmean cost is within 15% of the standard kmean at a negligible cpu  cost. 