# Time-Series Experiment

時系列のオープンデータを取得し、表データの特徴量に変換して機械学習モデルで予測する実験用リポジトリです。

## データ

既定では UCI Machine Learning Repository の `Individual Household Electric Power Consumption` を使います。

- 内容: 1世帯の電力消費量を約4年間、1分間隔で記録した多変量時系列
- 行数: 2,075,259
- タスク: 回帰、クラスタリング
- ライセンス: CC BY 4.0
- 出典: Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54

## セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## データ取得

```powershell
.\.venv\Scripts\python.exe fetch_open_data.py
```

`household_power_consumption.txt` が作成されます。

## 学習

```powershell
.\.venv\Scripts\python.exe power_forecasting.py --download-if-missing --model hgbt --freq 1h --forecast-horizon 24h --output-dir outputs
```

このスクリプトは次の流れで学習します。

1. 1分間隔の生データを `--freq` の間隔にリサンプリング
2. 時刻特徴量、周期特徴量、ラグ特徴量、移動平均・移動標準偏差を作成
3. 時系列順に train/test を分割
4. 指定モデルを学習し、MAE/RMSE を出力
5. テスト予測、将来予測、メトリクスを CSV/JSON で保存

利用できるモデルは次で確認できます。

```powershell
.\.venv\Scripts\python.exe power_forecasting.py --list-models
```
