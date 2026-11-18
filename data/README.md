# 데이터 저장 규칙

원본 데이터와 가공 데이터를 분리한다. 용량이 큰 데이터 파일은 GitHub에 직접 올리지 않고 출처와 재생성 방법을 기록한다.

| 폴더 | 저장 내용 | Git 추적 |
|---|---|---|
| `raw/` | 다운로드한 원본 JSON, CSV 또는 XLSX | 문서만 추적 |
| `interim/` | 시간당 한 행으로 변환하고 기본 정제를 마친 파일 | 문서만 추적 |
| `processed/` | 모델 학습에 사용하는 최종 데이터 | 문서만 추적 |
| `external/` | 기상, 공휴일 등 외부 보조 데이터 | 문서만 추적 |
| `sample/` | 코드 검증용 작은 합성 데이터 | 추적 |

## 원칙

- 원본 파일은 수정하거나 덮어쓰지 않는다.
- 모든 변환은 코드로 다시 실행할 수 있어야 한다.
- 실제 데이터와 합성 예제를 명확히 구분한다.
- 학습 데이터에는 생성 시각, 사용 기간, 포함 교차로, 행 수와 열 목록을 기록한 메타데이터를 함께 둔다.
- API 키나 인증 정보는 데이터 파일에 저장하지 않는다.

## 예상 원본 구조

| 열 | 의미 |
|---|---|
| `일자` | 관측 날짜 |
| `교차로명` | 스마트교차로 위치명 |
| `접근로명` | 진입 방향을 포함한 접근로 이름 |
| `합계` | 해당 날짜의 전체 교통량 |
| `00시`~`23시` | 시간별 교통량 |

실제 열 이름은 자료를 받은 뒤 확인한다. 코드가 임의로 추측하지 않도록 다른 열 이름은 명령행 옵션으로 지정한다.

## 파일명 규칙

월별 원본은 데이터 종류와 기간을 포함한다.

```text
cheonan_smart_intersection_traffic_2024-01.xlsx
cheonan_weather_2024-01.csv
```

## 확보한 실제 데이터

2026년 11월 15일 천안교통정보센터에서 2024년 1월부터 2026년 8월까지 전체 교차로 자료를 월별로 내려받았다. 공식 사이트의 엑셀 버튼이 호출하는 공개 JSON 응답을 원본 그대로 저장했다.

| 항목 | 확인 결과 |
|---|---|
| 원본 파일 | `raw/cheonan_smart_intersection_traffic_YYYY-MM.json` |
| 관측 기간 | 2024-01-01~2026-08-31 |
| 월별 파일 | 32개 |
| 원본 전체 행 | 221,922행 |
| 원본 전체 용량 | 136,695,198바이트 |
| 검증 | 모든 파일에 `resultList`가 있고 각 행의 날짜가 파일명 월과 일치 |

2024년 1월은 사용자가 직접 내려받은 XLSX도 함께 보관하고 기존 전처리 코드로 상세 품질을 확인했다.

| 항목 | 확인 결과 |
|---|---|
| 원본 파일 | `raw/cheonan_smart_intersection_traffic_2024-01.xlsx` |
| 관측 기간 | 2024-01-01~2024-01-31 |
| 원본 크기 | 6,262행, 28열 |
| 교차로 | 61개 |
| 교차로·접근로 조합 | 202개 |
| 중복 키 | 0행 |
| 결측값 | 0개 |
| 음수 교통량 | 0개 |
| 0 교통량 | 6,463개 셀 |
| 24시간 전체 0 | 1행 |
| 변환 결과 | `interim/cheonan_smart_intersection_traffic_2024-01_long.csv`, 150,288행 |

원본 XLSX의 SHA-256은 `54B736042A5D408B4FB03FA734574D1D2FCA1D238E30E35681EBCC2D4EA181EC`이다. 24시간 전체가 0인 행은 자동 삭제하지 않고 이후 EDA에서 실제 저교통량인지 센서 누락인지 확인한다.

## 모델링 대상 데이터

관측 연속성과 생활권 대표성을 확인하여 12개 교차로와 44개 접근로를 선정했다.

| 파일 | 내용 | 행 수 |
|---|---|---:|
| `interim/cheonan_traffic_wide_2024-01_2026-08.csv` | 32개월 전체 병합 데이터 | 221,922 |
| `processed/intersection_continuity.csv` | 교차로별 연속성·품질 지표 | 72 |
| `processed/approach_continuity.csv` | 접근로별 연속성·품질 지표 | 243 |
| `processed/selected_traffic_wide.csv` | 선정 교차로 와이드 데이터 | 42,856 |
| `processed/selected_traffic_long.csv` | 선정 교차로 시간별 롱 데이터 | 1,028,544 |
| `processed/eda/` | EDA 요약표와 재현용 수치 | 코드 실행 시 생성 |
| `processed/features/model_features.parquet` | 입력 변수·정답·시간순 분할 | 1,028,544 |
| `processed/features/high_traffic_thresholds.csv` | 접근로별 학습 구간 90백분위수 | 44 |

선정 목록과 근거는 `config/selected_intersections.csv`와 `docs/intersection_selection.md`에 기록한다.
EDA 결과와 그래프는 `docs/eda_report.md`에 기록하며 `python src/data/run_eda.py`로 다시 생성한다.
입력 변수 생성 결과는 `docs/feature_dataset.md`에 기록하며 `python src/features/build_features.py`로 다시 생성한다.

## 공식 출처

- 천안시 스마트교차로 교통량: https://www.data.go.kr/data/15160141/fileData.do
- 천안교통정보센터 통계: https://its.cheonan.go.kr/stat/sis-stat.do
- 기상청 시간별 기상자료: https://data.kma.go.kr

전체 데이터는 출처에서 다시 내려받고 `src/data`의 코드로 재생성할 수 있어야 한다.
