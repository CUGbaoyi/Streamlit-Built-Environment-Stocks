#!/bin/bash

# 查找空文件夹的函数
find_empty_dirs() {
    find "$1" -type d -empty -print0
}

# 在每个空文件夹中添加 .gitkeep 文件
find_empty_dirs . | while IFS= read -r -d '' dir; do
    touch "$dir/.gitkeep"
done

