#!/bin/bash

# 虚拟环境路径（修改为你的虚拟环境路径）
VENV_PATH="./venv/bin/activate"
source "$VENV_PATH"

# 数据集
test_dataset() {
   python3 -m unittest test.test_randomset
}

# 神经网络模型
test_model_nn() {
   python3 -m unittest test.test_perceptron
   python3 -m unittest test.test_ffnn
}

# 卷积神经网络
test_model_cnn() {
   python3 -m unittest test.test_cnn
}

# 主函数
main() {
    case "$1" in
        dataset)
            test_dataset
            ;;
        model_nn)
            test_model
            ;;
        model_cnn)
            test_model_cnn
            ;;
        *)
            echo "用法: $0 {dataset|model_nn|model_cnn}"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"