Fast API 기반 서버 사용

서버 실행시 uvicorn main:app --reload 터미널 입력
기본 포트 : 8000

/route/ 엔드포인트 POST 요청 시 
    start_node: int
    end_node: int
    day_of_week: str
    time_of_day: str
Body에 JSON 형식으로 입력