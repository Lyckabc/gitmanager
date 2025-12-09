import os
import subprocess
import requests
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# .env 파일 로드 (같은 폴더에 있는 .env 파일을 읽어옵니다)
load_dotenv()

app = FastAPI()

# -----------------------------------------------------------------------------
# 0. 헬퍼 함수: 환경변수가 있으면 기본값으로, 없으면 필수 필드로 설정
# -----------------------------------------------------------------------------
def env_default(key: str, default_val=None):
    """
    .env에서 값을 가져옵니다.
    - 값이 있으면 그 값을 기본값(default)으로 설정합니다. (API Body에서 생략 가능)
    - 값이 없고 별도 기본값이 없으면 Pydantic의 '...' (Ellipsis)를 반환하여 필수 필드로 만듭니다.
    """
    val = os.getenv(key)
    if val:
        return val
    if default_val is not None:
        return default_val
    return ...  # 필수 필드 (Required) 표시

# -----------------------------------------------------------------------------
# 1. 데이터 모델 정의 (Pydantic) - .env 연동
# -----------------------------------------------------------------------------
class PRRequest(BaseModel):
    # GitHub 정보
    owner: str = Field(default=env_default("GITHUB_OWNER"))
    repository: str = Field(default=env_default("GITHUB_REPOSITORY"))
    token: str = Field(default=env_default("GITHUB_TOKEN"))
    
    # Branch는 자주 바뀌므로 env가 없으면 필수로 받거나, 기본값을 main으로 둘 수 있음
    branch: str = Field(default=env_default("GITHUB_BRANCH"), description="PR을 생성할 소스 브랜치")
    base_branch: str = Field(default=env_default("GITHUB_BASE_BRANCH", "dev"))

    # Git Log 설정
    n_commits: int = Field(default=os.getenv("GIT_N_COMMITS", 14))
    
    # PR 내용 설정
    pr_template: str = Field(default=env_default("PR_TEMPLATE", "## 변경 사항\n- \n\n## 리뷰 포인트\n- "))
    
    # LLM 설정
    llm_provider: Literal["gemini", "gpt"] = Field(default=env_default("LLM_PROVIDER"))
    llm_api_key: str = Field(default=env_default("LLM_API_KEY"))
    llm_model: str = Field(default=env_default("LLM_MODEL", "gemini-2.0-flash-lite"))

# -----------------------------------------------------------------------------
# 2. 헬퍼 함수: Git 명령어 실행
# -----------------------------------------------------------------------------
def run_git_commands(n: int, base: str, branch: str) -> str:
    try:
        # Fetch from origin to make sure we have the latest branches
        subprocess.check_output("git fetch origin", shell=True)

        # Sanitize branch name, removing "origin/" if it exists
        local_branch = branch
        if local_branch.startswith("origin/"):
            local_branch = local_branch[len("origin/"):]

        # Checkout the branch to ensure HEAD is correct
        subprocess.check_output(f"git checkout {local_branch}", shell=True)

        # 1. 기준 커밋 해시 찾기
        rev_list_cmd = f"git rev-list --reverse {base}..HEAD | tail -n {n} | head -n 1"
        target_commit = subprocess.check_output(rev_list_cmd, shell=True).decode().strip()
        
        if not target_commit:
            raise Exception("변경사항을 찾을 수 없거나 커밋 범위가 유효하지 않습니다.")

        # 2. Diff, Log, Stat 추출
        diff_cmd = f"git diff {target_commit}^..HEAD"
        log_cmd = f'git log {base}..HEAD -n {n} --pretty=format:"%h %s" --name-status'
        stat_cmd = f"git diff --stat {target_commit}^..HEAD"

        changes_diff = subprocess.check_output(diff_cmd, shell=True).decode()
        commit_summary = subprocess.check_output(log_cmd, shell=True).decode()
        changes_stat = subprocess.check_output(stat_cmd, shell=True).decode()

        full_context = f"""
=== COMMIT SUMMARY ===
{commit_summary}

=== CHANGES STAT ===
{changes_stat}

=== GIT DIFF (Detail) ===
{changes_diff[:10000]} 
""" 
        return full_context

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Git command failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# 3. 헬퍼 함수: LLM API 호출
# -----------------------------------------------------------------------------
def generate_pr_content(provider: str, api_key: str, model: str, template: str, git_data: str) -> dict:
    
    system_prompt = "You are an expert software developer and technical writer. Your task is to refine a Pull Request (PR) template and then draft a PR description using that refined template, referencing provided file changes."
    user_prompt = f"""
Please write a Pull Request body.

[PR Template]
{template}

[Code Changes & Logs]
{git_data}

Fill out the template based on the changes. Keep the tone professional.
"""

    if provider == "gpt":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"LLM Error: {response.text}")
        
        content = response.json()["choices"][0]["message"]["content"]
        return {"title": f"Update: {model} generated PR", "body": content}

    elif provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": system_prompt + "\n" + user_prompt}]}]}
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"LLM Error: {response.text}")
        
        try:
            content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return {"title": "Automated PR by Gemini", "body": content}
        except KeyError:
             raise HTTPException(status_code=500, detail=f"Gemini response parsing failed: {response.text}")

    else:
        raise HTTPException(status_code=400, detail="Unsupported LLM provider")

# -----------------------------------------------------------------------------
# 4. 헬퍼 함수: GitHub PR 생성
# -----------------------------------------------------------------------------
def create_github_pull_request(req: PRRequest, title: str, body: str):
    url = f"https://api.github.com/repos/{req.owner}/{req.repository}/pulls"
    headers = {"Authorization": f"token {req.token}", "Accept": "application/vnd.github.v3+json"}
    payload = {"title": title, "body": body, "head": req.branch, "base": req.base_branch}
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 201:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=f"GitHub Error: {response.text}")

# -----------------------------------------------------------------------------
# 5. 메인 Endpoint
# -----------------------------------------------------------------------------
@app.post("/create-pr")
def create_pr_endpoint(request: PRRequest):
    # 1. Git 데이터 수집
    git_data = run_git_commands(request.n_commits, request.base_branch, request.branch)
    
    # 2. LLM PR 생성
    generated_content = generate_pr_content(
        provider=request.llm_provider,
        api_key=request.llm_api_key,
        model=request.llm_model,
        template=request.pr_template,
        git_data=git_data
    )
    
    # 3. GitHub PR 전송
    pr_result = create_github_pull_request(
        req=request,
        title=generated_content["title"],
        body=generated_content["body"]
    )
    
    return {
        "status": "success",
        "pr_url": pr_result.get("html_url"),
        "pr_number": pr_result.get("number")
    }

if __name__ == "__main__":
    import uvicorn
    if not os.path.exists(".git"):
        print("Warning: .git directory not found. Run inside a git repo.")
    uvicorn.run(app, host="0.0.0.0", port=8000)