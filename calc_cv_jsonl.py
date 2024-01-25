import jsonlines
import numpy as np
import csv

# JSONLファイルのパス
jsonl_file_path = 'logdir/run1/metrics_eval_2.5M_cv.jsonl'

# スコアを格納するリスト
scores = []

# JSONLファイルを開いて読み込む
with jsonlines.open(jsonl_file_path) as reader:
    for line in reader:
        if 'episode/score' in line:
            scores.append(line['episode/score'])

# 平均と分散を計算
average_scores = []
std_scores = []
cv_scores = []
for i in range(len(scores)):
    average_scores.append(np.mean(scores[:i+1]))
    std_scores.append(np.std(scores[:i+1]))
    cv_scores.append(std_scores[-1] / average_scores[-1])
print("cv:", cv_scores)

# CSVファイルに書き込み
csv_file_path = 'scores_data.csv'
with open(csv_file_path, mode='w', newline='') as csv_file:
    fieldnames = ['Episode', 'Average_Score', 'Standard_Deviation', 'Coefficient_of_Variation']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(scores)):
        writer.writerow({
            'Episode': i + 1,
            'Average_Score': average_scores[i],
            'Standard_Deviation': std_scores[i],
            'Coefficient_of_Variation': cv_scores[i]
        })

# # 結果を出力
# print(f'データ数: {len(scores)}')
# print(f'平均スコア: {average_score}')
# print(f'スコアの標準偏差: {std_score}')
# print(f'スコアの変動係数: {cv_score}')