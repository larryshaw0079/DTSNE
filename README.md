# Introduction

Official implementation of paper "Unsupervised Anomaly Detection with Distillated Teacher-Student Network Ensemble".

Full text: https://www.mdpi.com/1099-4300/23/2/201/htm

![image-20210914232111430](https://i.loli.net/2021/09/14/MkqFTsWo8I1lBKH.png)

![image-20210914232132535](https://i.loli.net/2021/09/14/tX847mNyxHSjTfF.png)

# Usage

A typical command that performs 10 independent runs would be:

`python main.py --data-path <your-data-path> --preprocessing minmax --seed 2020 --num-students 8 --num-trans 32 --num-trial 10 --pretrain --classify-score cp --batch-size 128`

To see more options, please type `python main.py -h`.

# Citation

If you find our paper is helpful for your research, please cite this paper:

```
@Article{e23020201,
	AUTHOR = {Xiao, Qinfeng and Wang, Jing and Lin, Youfang and Gongsa, Wenbo and Hu, Ganghui and Li, Menggang and Wang, Fang},
	TITLE = {Unsupervised Anomaly Detection with Distillated Teacher-Student Network Ensemble},
	JOURNAL = {Entropy},
	VOLUME = {23},
	YEAR = {2021},
	NUMBER = {2},
	ARTICLE-NUMBER = {201},
	URL = {https://www.mdpi.com/1099-4300/23/2/201},
	PubMedID = {33561954},
	ISSN = {1099-4300},
	DOI = {10.3390/e23020201}
}
```

