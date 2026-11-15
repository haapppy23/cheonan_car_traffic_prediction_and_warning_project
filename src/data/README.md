# data code

- `download_cheonan_traffic.py`: 공식 사이트의 공개 응답을 30일 제한에 맞춰 월별 JSON으로 내려받고 기간을 검증한다.
- `prepare_monthly_traffic.py`: 월별 JSON을 하나의 와이드 데이터로 병합하고 접근로·교차로 연속성 보고서를 만든다.
- `select_intersections.py`: 선정 목록과 연속성 지표를 검증하고 선정 교차로의 와이드·롱 데이터를 만든다.
- `run_eda.py`: 선정된 시간별 자료의 품질·시간대·요일·월별 패턴을 분석하고 보고서와 그래프를 만든다.
- `inspect_raw.py`: 원본 열 구조, 결측, 중복, 음수와 0값을 검사한다.
- `wide_to_long.py`: 하루 24시간 열을 시간당 한 행으로 변환한다.
- `schema.py`: 두 스크립트가 공유하는 열 탐지와 파일 입출력 규칙을 관리한다.

실제 열 이름이 예상과 다르면 명령행의 열 이름 옵션을 사용한다. 코드가 알 수 없는 열을 조용히 추측하지 않는다.

```bash
python src/data/download_cheonan_traffic.py --start-month 2024-01 --end-month 2026-08
python src/data/prepare_monthly_traffic.py
python src/data/select_intersections.py
python src/data/run_eda.py
```

이미 존재하는 월 파일은 응답 구조와 기간을 다시 검사한 뒤 건너뛴다. 다시 내려받아야 할 때만 `--force`를 사용한다.
