# TPU-Benchmark
TPU benchmarks in colaboratory.

# Models
103M-params WideResNet(k=7, N=4). Changed from 3x3 conv to 7x7 conv.

```
Total params: 103,465,028
Trainable params: 103,452,484
Non-trainable params: 12,544
```

# Result
Compare the average of second per epoch after 2nd epoch.

## Linear scale
![](https://github.com/koshian2/TPU-Benchmark/blob/master/images/01_batch_size.png)

## Log scale
![](https://github.com/koshian2/TPU-Benchmark/blob/master/images/02_batch_size_log.png)

[More results:](https://github.com/koshian2/TPU-Benchmark/tree/master/images)

# More details(Japanese)
[https://qiita.com/koshian2/items/fb989cebe0266d1b32fc](https://qiita.com/koshian2/items/fb989cebe0266d1b32fc)
