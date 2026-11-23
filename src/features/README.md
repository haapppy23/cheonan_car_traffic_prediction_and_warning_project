# features

시간 특성, 지연 변수, 이동 통계와 외부 데이터 결합 코드를 둔다.

- `build_features.py`: 선정된 시간별 교통량으로 모델 입력 변수, 정답, 고교통량 기준과 시간순 분할을 생성한다.
- `build_external_features.py`: 대한민국 공휴일과 천안 시간별 기상 자료를 기존 모델 행에 결합한다.

```bash
python src/features/build_features.py
python src/data/download_cheonan_weather.py
python src/features/build_external_features.py
```

외부 자료를 추가한 파일은 `model_features_external.parquet`로 따로 저장하여 기존 최종 모델 입력을 덮어쓰지 않는다.
