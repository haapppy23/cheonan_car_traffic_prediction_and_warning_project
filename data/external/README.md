# external

기상, 공휴일 등 외부 보조 데이터를 저장한다. 출처와 수집 시점을 기록하고 실제 파일은 Git으로 추적하지 않는다.

시간별 기상은 Open-Meteo Historical Weather API에서 천안 중심 좌표(36.8151, 127.1139)의 기온·강수량·10m 풍속을 내려받는다. 이 값은 교차로 현장 관측값이 아니라 재분석 격자 값이다.

```bash
python src/data/download_cheonan_weather.py
```

- 공식 문서: https://open-meteo.com/en/docs/historical-weather-api
- 생성 파일: `cheonan_weather_hourly.csv`, `cheonan_weather_metadata.json`
- 사용 기간: 2024-01-01~2026-08-31
- 시간대: Asia/Seoul

대한민국 공휴일 여부는 별도 파일을 내려받지 않고 `holidays` 패키지의 KR 달력을 이용해 `src/features/build_external_features.py`에서 재현한다.
