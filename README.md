### 2. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화 (Linux/Mac)
source venv/bin/activate

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# library 설치
python3 -m pip install -r requirements.txt

# 8011 포트 fastAPI 실행
uvicorn main:app --port 8012 --reload
```
