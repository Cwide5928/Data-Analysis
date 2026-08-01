# 반도체 공정 센서 기반 불량(Yield) 예측 및 원인 센서 분석
590개 공정 센서로 웨이퍼 합불을 예측하고, 시각 탐색(Spotfire)과 머신러닝(Python)을 교차검증해 불량 연관 센서를 도출한 프로젝트. 극단적 클래스 불균형·결측 처리와 SPC 관점 해석.
> **핵심 결과** — 개별 센서로는 판별 불가 → RF(AUC 0.80)+임계값 조정으로 불량 62% 검출, 상위 센서 sensor_59·103·33 도출.

## 1. Project Overview
- 목표: 공정 중 측정된 센서 값만으로 최종 합불(Pass/Fail)을 예측하고, 불량과 연관된 센서를 좁혀내기 위함. (Pass/Fail은 센서가 계산한 값이 아니라 별도 최종 검사의 실제 결과이며, 목표는 센서로 그 결과를 예측·선별하고 원인을 추적하는 것)
- JD 연결: 수율/데이터 분석 직무 우대사항 "데이터 시각화·머신러닝 분석", "통계적 공정관리(SPC) 기반 품질 데이터 분석"에 대응.


## 2. Data
- UCI SECOM (Kaggle uci-secom.csv)의 1,567 웨이퍼 × 590 익명 센서 + 타임스탬프 + Pass/Fail(양품 1,463 / 불량 104). 센서는 익명화되어 물리적 의미 비공개. 2008년 공개 벤치마크(방법론 학습용).


## 3. 사용 도구
- Python: pandas, numpy, scikit-learn (Pipeline, RandomForest, LogisticRegression), matplotlib

- 시각화: Spotfire (SPC 관리도·분포 탐색), Jupyter Notebook


## 4. 분석 흐름
- [1] Spotfire를 활용한 시각적 탐색 — "센서 하나로 갈리는가" 막대(불균형 6.6%) → 상자그림·산점도(단일·이변량 센서로 판별 불가) → SPC 관리도(±3σ, OOC ≠ 불량 확인) → 히스토그램(우측 치우침 → 중앙값이 강건). 결론: 개별 센서로는 합불이 안 갈린다 → 다변량 머신러닝 필요.

- [2] Python 머신러닝 — "어떤 센서가 원인인가" 전처리(결측 50%↑·상수 센서 제거, 중앙값 대치, 표준화, 층화 분할) → 기준 모델(LogReg) vs Random Forest → precision-recall 곡선으로 임계값 조정 → Feature Importance.

- [3] 교차검증 — 앞선 결과를 바탕으로 영향이 가장 큰 센서(sensor_59)를 Spotfire 상자그림으로 재확인: 양품 중앙값 0.72 vs 불량 5.52로 실제 분포 차이 확인(겹쳤던 sensor_0와 대조).

<img width="1434" height="960" alt="image" src="https://github.com/user-attachments/assets/e045d6cd-2295-44c7-b5e6-f1d466168223" />


## 5. 성능 요약
| 실험  | 모델                  | 불균형 처리                | 임계값      | Recall(불량) | Precision(불량) | ROC-AUC | 비고                                                 |
| --- | ------------------- | --------------------- | -------- | ---------- | ------------- | ------- | -------------------------------------------------- |
| 01  | Logistic Regression | class_weight=balanced | 0.5      | 0.19       | 0.10          | 0.669   | 기준 모델. accuracy 0.83이 "전부 양품"(93.4%)보다 낮음 = 정확도 함정 |
| 02  | Random Forest       | class_weight=balanced | 0.5      | 0.00       | 0.00          | 0.798   | 순위 능력↑, but 0.5 임계값에선 전부 양품 예측                     |
| 03  | Random Forest       | class_weight=balanced | **0.09** | **0.62**   | 0.20          | 0.798   | 임계값 조정으로 불량 62% 검출 (26개 중 약 16개)                   |

정확도가 아니라 Recall·ROC-AUC를 기준으로 평가. 불량 유출(소비자 위험)이 오검(생산자 위험)보다 치명적이므로 Recall 중심으로 임계값을 낮춰 튜닝. 최종 임계값은 비용 등 다양한 상황을 고려하여 결정해야함

<img width="473" height="498" alt="image" src="https://github.com/user-attachments/assets/4962d421-18ae-4531-9f16-641d2f971a44" />

*LogReg(AUC 0.67) vs RF(AUC 0.80). RF가 좌상단에 더 붙어 순위 능력 우수.*


## 6. Key Findings
- 개별 센서로는 합불 판별 불가(시각 탐색), 590개를 조합하는 모델은 순위 능력(AUC 0.80)을 가지나 기본 임계값(0.5)에선 의미 없음 → 모델이 아니라 임계값이 문제임을 실증.
- 임계값을 0.09로 조정해 불량의 62% 검출(정밀도 20%).
- Feature Importance: 불량 연관 상위 sensor_59(0.016), sensor_103(0.013), sensor_33(0.012). 단 중요도가 낮고 고르게 분포 → 단일 원인 센서는 없으며 불량은 다변량 상호작용의 결과.
- 모델이 지목한 sensor_59가 시각(상자그림)에서도 실제 분포 차이를 보여 모델↔시각화 교차검증 성립.

<img width="594" height="780" alt="image" src="https://github.com/user-attachments/assets/f3e92922-aa31-49d6-b3c0-038bb535305f" />



## 7. How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
# uci-secom.csv 를 노트북과 같은 폴더에 두기 (Kaggle: uciml/secom)
jupyter notebook secom_yield_prediction.ipynb   # 위에서 아래로 순서대로 실행
```


## 8. 한계
- 센서 익명화로 물리적 원인 해석 불가.
- 데이터는 2008년 공개 벤치마크로 최신 공정이 아닌 방법론 학습용.
- 규격이 없어 실제 Cpk 미산출.


## 9. 배운 점 / 역량

- 통계적 공정관리(관리한계 vs 규격한계, OOC)
- 불균형 데이터 평가(정확도 함정, Recall·Precision·ROC-AUC·PR curve, 소비자/생산자 위험)
- 임계값 튜닝과 비용 기반 의사결정
- 데이터 정제(헤더 유입·결측·이상치)
- Random Forest·Feature Importance
- 모델↔시각화 교차검증

## 학습 과정
- LLM을 학습 튜터로 활용하되, 모든 개념을 1차 자료(공식 문서·실제 사례)로 교차검증하고, 모델 결과를 Spotfire 시각화로 재검증하는 방식으로 진행함.

Data: Kaggle SECOM Dataset
