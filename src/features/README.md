# features

시간 특성, 지연 변수, 이동 통계와 외부 데이터 결합 코드를 둔다.

- `build_features.py`: 선정된 시간별 교통량으로 모델 입력 변수, 정답, 고교통량 기준과 시간순 분할을 생성한다.

```bash
python src/features/build_features.py
```

공휴일과 기상 자료는 외부 출처를 확보한 뒤 별도 결합한다. 현재 코드는 확인되지 않은 값을 임의로 채우지 않는다.
