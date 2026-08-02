@echo off
chcp 65001
REM One-click infra stop: Kibana, Elasticsearch, Milvus containers, Docker.
REM Usage: double-click or run from terminal: scripts/infra_stop.bat
REM Start: scripts/infra_stop.bat
REM Companion: scripts/infra_start.bat

REM Check admin privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在请求管理员权限...
    powershell -Command "Start-Process "%~f0" -Verb RunAs"
    exit /b
)

echo 正在停止基础设施服务...

echo [1/4] 停止 Kibana...
sc stop kibana >nul 2>&1

echo [2/4] 停止 Elasticsearch...
sc stop elasticsearch-service-x64 >nul 2>&1

echo [3/4] 停止 Milvus 容器...
docker stop milvus-standalone >nul 2>&1
docker stop milvus-minio >nul 2>&1
docker stop milvus-etcd >nul 2>&1

echo [4/4] 停止 Docker 服务...
sc stop com.docker.service >nul 2>&1

echo.
echo Kibana：         已停止
echo Elasticsearch：  已停止
echo Milvus 容器：    已停止
echo Docker：         已停止
echo.
echo 所有基础设施服务已停止。
pause
