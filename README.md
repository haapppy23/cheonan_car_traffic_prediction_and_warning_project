# 천안시 스마트교차로 교통량 예측 및 고교통량 조기경보

천안시 스마트교차로의 시간별 교통량을 이용해 접근로별 1시간 후 교통량을 예측하고, 해당 접근로의 평소 상위 10% 수준을 넘는 고교통량 위험을 알리는 프로젝트이다.

## 프로젝트 목표

- 주 예측 대상: 교차로 접근로별 1시간 후 차량 통행량
- 보조 결과: 학습 기간의 접근로별 교통량 90백분위수 초과 여부
- 활용 대상: 시민의 이동 시간 조정과 교통 관리자의 집중 관찰 지점 선정
- 해석 범위: 속도와 통행시간 자료가 없으므로 실제 정체를 확정하지 않고 고교통량 위험으로 표현한다.

## 사용 예정 데이터

| 데이터 | 용도 |
|---|---|
| 천안시 스마트교차로 교통량 | 주 학습 데이터 |
| 기상청 시간별 기상자료 | 기온, 강수량, 풍속 후보 변수 |
| 공휴일 정보 | 평일, 주말, 공휴일 구분 |

실제 데이터 파일은 저장소에 포함하지 않는다. 데이터 출처와 저장 규칙은 [data/README.md](data/README.md)에 기록한다.

## 전체 처리 흐름

```text
월별 원본 파일
→ 원본 구조와 품질 검사
→ 와이드 형식을 시간당 한 행의 롱 형식으로 변환
→ 사용할 교차로 선정과 EDA
→ 시간·과거 교통량·날씨 입력 변수 생성
→ 시간순 학습·검증·시험 데이터 구성
→ 단순 기준과 여러 후보 모델 비교
→ 최종 모델 저장
→ FastAPI와 Streamlit 연결
```

## 프로젝트 구조

```text
data/                 데이터 저장 규칙과 작은 합성 예제
notebooks/            EDA와 실험 노트북
src/data/             원본 검사와 전처리 코드
src/features/         입력 변수 생성 코드
src/models/           학습 코드
src/evaluation/       지표와 오류 분석 코드
models/               저장된 모델 파일 위치
backend/              FastAPI 서버
frontend/             Streamlit 화면
tests/                자동 테스트
docs/                 데이터·실험 설계와 담당자 설명서
```

## 현재 구현된 범위

- 프로젝트와 데이터 폴더 구조
- 합성 와이드 예제 데이터
- 2024년 1월~2026년 8월 월별 실제 교통량 다운로드와 검증
- CSV·XLSX 원본 구조 검사
- 와이드 데이터를 시간당 한 행의 롱 형식으로 변환
- 전체 자료 병합과 교차로·접근로 관측 연속성 계산
- 4개 생활권의 모델링 대상 교차로 12개 선정
- 선정 자료 1,028,544행의 EDA와 품질 플래그 생성
- 생활권·시간대·요일·월별 패턴과 교차로별 0값 분석
- 과거 교통량·시간 특성·이동 통계 입력 변수 생성
- 예측 대상 시각 기준의 학습·검증·시험 분할
- 학습 구간만 사용한 접근로별 고교통량 기준 생성
- 검증 구간의 지속성·전날·전주 단순 기준 비교
- 고정 Ridge 회귀 학습과 단순 기준 대비 검증
- 고정 HistGradientBoosting 학습과 Ridge 대비 검증
- 접근로별 기준을 적용한 고교통량 경보 지표 검증
- 전처리·EDA·입력 변수 자동 테스트
- 데이터·실험 설계 문서와 데이터/ML 담당자 설명서

아직 다른 후보 모델과 세부 오류 분석, 최종 모델 선정, FastAPI와 Streamlit 구현은 수행하지 않았다.

## 설치

Python 3.11 이상 환경을 권장한다.

```bash
python -m pip install -r requirements.txt
```

## 실행

합성 원본 구조를 검사한다.

```bash
python src/data/inspect_raw.py --input data/sample/cheonan_traffic_wide_sample.csv
```

합성 와이드 데이터를 롱 형식으로 변환한다.

```bash
python src/data/wide_to_long.py --input data/sample/cheonan_traffic_wide_sample.csv --output data/interim/cheonan_traffic_long_sample.csv
```

선정된 실제 자료의 EDA 보고서와 그래프를 다시 생성한다.

```bash
python src/data/run_eda.py
```

모델 입력 변수와 시간순 분할을 생성한다.

```bash
python src/features/build_features.py
```

시험 구간을 열지 않고 검증 구간의 단순 기준을 평가한다.

```bash
python src/evaluation/run_baselines.py
```

고정 Ridge 회귀를 학습하고 단순 기준과 비교한다.

```bash
python src/models/train_ridge.py
```

고정 HistGradientBoosting을 학습하고 Ridge와 비교한다.

```bash
python src/models/train_hist_gradient_boosting.py
```

최고 회귀 모델의 고교통량 경보 성능을 평가한다.

```bash
python src/evaluation/run_high_traffic_alerts.py
```

자동 테스트를 실행한다.

```bash
python -m pytest
```

실제 원본의 열 이름이 예상과 다르면 명령행에서 명시적으로 지정할 수 있다.

```bash
python src/data/inspect_raw.py --input 원본.xlsx --date-column 날짜 --intersection-column 교차로 --approach-column 진입로
```

## 팀 역할

| 담당 | 역할 |
|---|---|
| 김성현 | 데이터 검사·전처리·EDA·공통 학습 파이프라인·최종 모델 비교 |
| 김문일 | Streamlit 프론트엔드와 배정된 후보 모델 실험 |
| 이나율 | FastAPI 백엔드와 배정된 후보 모델 실험 |

각 역할의 실제 기여는 작성한 코드, 문서, 테스트와 Git 커밋으로 남긴다.

## Git에 데이터를 올리지 않는 이유

원본과 가공 데이터는 용량이 커질 수 있고 데이터 출처의 이용 조건을 확인해야 한다. 모델 파일과 API 키도 코드와 분리해야 한다. 저장소에는 작은 합성 예제, 데이터 출처, 재생성 코드와 실행 방법만 포함한다.

## 다음 단계

현재 최고 모델의 교차로·시간대별 회귀 오차와 놓친 고교통량 사례를 분석한다. 경보 기준점은 Precision 0.815, Recall 0.680, F1 0.742, PR-AUC 0.815이다.
