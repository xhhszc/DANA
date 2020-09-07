# Domain-adversarial Network Alignment
This is the codes and data for DANA model in our paper "[Domain-adversarial Network Alignment][1]".

If you would like to acknowledge our efforts, please cite the following paper:

    @article{hong2020dana,
    title={https://arxiv.org/abs/1908.05429},
    author={Hong, Huiting and Li, Xin and Pan, Yuangang and W. Tsang, Ivor},
    journal={IEEE Transactions on Knowledge and Data Engineering},
    year={2020},
    publisher={IEEE}
    }

# Prerequisites

python 2.7

tensorflow >= 1.9

networkx == 2.2



# How to run
```shell
cd DANA/gcn
sh sh_train.sh
```



## Note
- The output files (learned embeddings) will be stored in the ``output`` directory during training process.



[1]: https://arxiv.org/abs/1908.05429