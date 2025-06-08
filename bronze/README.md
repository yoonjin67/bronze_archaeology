# WARNING
반드시 파이썬 스크립트는 현재 경로에서 실행해야 합니다.
파이썬 버전은 3.12, 아나콘다 등으로 설치 후 requirements.txt를 이용해 패키지를 설치해 주십시오.


# Project Tree
```bash
.
├── README.md
├── bronze.csv
├── results
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── dbscan_bronze.png
│   ├── hdbscan_bronze.png
│   ├── kmeans_bronze.png
│   ├── logistic_regression_bronze.png
│   └── ransom_forest.png
├── supervised
│   ├── logistic
│   │   └── logistic_regression_bronze.py
│   └── random_forest
│       └── random_forest_bronze.py
└── unsupervised
    ├── dbscan
    │   └── dbscan_chinese_bronze.py
    ├── hdbscan
    │   └── hdbscan_chinese_bronze.py
    └── kmeans
        └── kmeans_chinese_bronze.py

9 directories, 14 files

```

