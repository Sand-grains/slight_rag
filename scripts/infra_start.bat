@echo off
chcp 65001
REM One-click infra start: Docker (Milvus), Elasticsearch, Kibana.
REM Usage: double-click or run from terminal: scripts/infra_start.bat
REM Start: scripts/infra_start.bat
REM Companion: scripts/infra_stop.bat

REM Check admin privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在请求管理员权限...
    powershell -Command "Start-Process "%~f0" -Verb RunAs"
    exit /b
)

echo 正在启动基础设施服务...

echo [1/5] 启动 Docker 服务...
sc start com.docker.service >nul 2>&1

echo [*] 等待 Docker 守护进程就绪（最多 30 秒）...
for /L %%i in (1,1,30) do (
    docker info >nul 2>&1 && goto :docker_ready
    ping 127.0.0.1 -n 2 >nul 2>&1
)
echo 错误：Docker 守护进程 30 秒内未就绪
pause
exit /b 1

:docker_ready
echo [2/5] 启动 Milvus 容器...
docker start milvus-etcd >nul 2>&1
docker start milvus-minio >nul 2>&1
docker start milvus-standalone >nul 2>&1
echo       etcd, minio, standalone 已启动

echo [3/5] 启动 Elasticsearch...
sc start elasticsearch-service-x64 >nul 2>&1

echo [4/5] 启动 Kibana...
sc start kibana >nul 2>&1

echo.
echo Docker（Milvus）：已启动
echo Elasticsearch：  启动中...
echo Kibana：         启动中...
echo.
echo 等待 30 秒，确保服务就绪...
ping 127.0.0.1 -n 31 >nul 2>&1
echo.
echo 所有服务应已就绪，请验证：
echo   Milvus 管理界面：http://localhost:9091/webui/
echo   Kibana 控制台：  http://localhost:5601
pause
