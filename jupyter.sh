#!/bin/bash

# Jupyter 日志文件路径
LOG_FILE="./logs/jupyter.log"

# Jupyter 服务端口
PORT=8888

# 虚拟环境路径（修改为你的虚拟环境路径）
VENV_PATH="./venv/bin/activate"

# 启动 Jupyter 服务
start_jupyter() {
    if is_jupyter_running; then
        echo "Jupyter 服务已经在运行中。"
    else
        echo "正在启动 Jupyter 服务..."
        # 激活虚拟环境
        source "$VENV_PATH"
        nohup jupyter notebook --port="$PORT" > "$LOG_FILE" 2>&1 &
        echo "Jupyter 服务已启动，日志文件: $LOG_FILE"
    fi
}

# 停止 Jupyter 服务
stop_jupyter() {
    PID=$(get_jupyter_pid)
    if [ -z "$PID" ]; then
        echo "Jupyter 服务未运行。"
    else
        echo "正在停止 Jupyter 服务 (PID: $PID)..."
        kill "$PID"
        echo "Jupyter 服务已停止。"
    fi
}

# 查看 Jupyter 服务是否在运行
status_jupyter() {
    if is_jupyter_running; then
        echo "Jupyter 服务正在运行。"
    else
        echo "Jupyter 服务未运行。"
    fi
}

# 获取 Jupyter 服务的进程 ID
get_jupyter_pid() {
    pgrep -f "jupyter-notebook"
}

# 检查 Jupyter 服务是否在运行
is_jupyter_running() {
    if [ -z "$(get_jupyter_pid)" ]; then
        return 1
    else
        return 0
    fi
}

# 主函数
main() {
    case "$1" in
        start)
            start_jupyter
            ;;
        stop)
            stop_jupyter
            ;;
        status)
            status_jupyter
            ;;
        *)
            echo "用法: $0 {start|stop|status}"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"