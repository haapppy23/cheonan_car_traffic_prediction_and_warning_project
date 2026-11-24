# FastAPI 예측 서버

최종 Random Forest와 저장된 시간별 교통량을 불러와 접근로별 1시간 후 교통량과 고교통량 위험을 반환한다. 사용자가 지연 변수나 이동 평균을 직접 입력하지 않으며, 서버가 예측 직전 168시간 이력으로 모델 입력을 만든다.

## 실행 전 준비

다음 세 파일이 로컬에 있어야 한다.

- `models/final_random_forest.joblib`
- `data/processed/selected_traffic_long.csv`
- `data/processed/features/high_traffic_thresholds.csv`

최종 모델 파일이 없으면 프로젝트 루트에서 다음 명령으로 생성한다.

```bash
python src/models/finalize_random_forest.py
```

## 서버 실행

프로젝트 루트에서 실행한다.

```bash
python -m uvicorn backend.app:app --reload
```

실행 후 `http://127.0.0.1:8000/docs`에서 Swagger 화면으로 요청을 시험할 수 있다.

## 상태 확인

```http
GET /health
```

```json
{
  "status": "ok",
  "model": "random_forest"
}
```

## 교통량 예측

```http
POST /predict
Content-Type: application/json
```

```json
{
  "intersection": "도로원점3",
  "approach": "도로원점3_남"
}
```

`target_timestamp`를 생략하면 해당 접근로에 저장된 마지막 관측의 1시간 후를 예측한다. 과거 시각을 재현할 때는 시간대 표기 없는 한국 현지 정각을 넣는다.

```json
{
  "intersection": "도로원점3",
  "approach": "도로원점3_남",
  "target_timestamp": "2026-09-01T00:00:00",
  "predicted_volume": 193.502,
  "high_traffic_threshold": 1898.0,
  "high_traffic_alert": false,
  "risk_score": 0.102,
  "major_factors": [
    {
      "feature": "lag_1",
      "label": "1시간 전 교통량",
      "contribution": -177.43,
      "effect": "decrease"
    }
  ]
}
```

저장되지 않은 교차로·접근로는 `404`, 예측 직전 168시간 이력이 완전하지 않으면 `422`를 반환한다. `high_traffic_alert`는 정체 확정이 아니라 평소 상위 10% 수준을 넘는 교통량 위험을 뜻한다.

`major_factors`는 해당 예측에 대한 Tree SHAP 절댓값 상위 세 변수다. `contribution`은 예상 교통량을 높이거나 낮춘 모델 내부 기여량이며 실제 인과관계를 뜻하지 않는다.

### GET /locations
사용 가능한 교차로 및 접근로 목록을 반환합니다.

---

**응답 예시:**
```json
{
  "locations": [
    {
      "intersection": "도로원점3",
      "approaches": [
        "도로원점3_남",
        "도로원점3_동"
      ]
    }
  ]
}
```
