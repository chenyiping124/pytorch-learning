#!/bin/bash

# 虚拟环境路径（修改为你的虚拟环境路径）
VENV_PATH="./venv/bin/activate"
source "$VENV_PATH"

# 数据集
test_dataset() {
   python3 -m unittest test.test_randomset
}

# 模型
test_model() {
   python3 -m unittest test.test_perceptron
   python3 -m unittest test.test_ffnn
}

# 主函数
main() {
    case "$1" in
        dataset)
            test_dataset
            ;;
        model)
            test_model
            ;;
        *)
            echo "用法: $0 {dataset|model}"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"