# XGBoost model tracking with MLflow
made a machine learning model using XGBoost for the dataset called "spaceship_titanic" uploaded on Kaggle.  
model results are logged onto MLflow.  
I uploaded two files named "xgboost.ipynb" and "xgboost_with_ctgan.ipynb". the former is just a xgboost model learning original spaceship_titanic data, the later uses CT-GAN to create synthesized data set from the original data, and learn xgboost with it.  

クラウド上へのデータアップロードをトリガーとして、モデル学習とその学習結果の管理・可視化を一気通貫で行うシステムを作成しました。  
AWS上でストレージやサーバを管理しその中でMLflowを実装することで、すべてをクラウド上で実行管理できるようになり、モデル構築や運用に際しての大幅な効率化を実現しています。
