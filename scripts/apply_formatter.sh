#!/bin/bash
set -eux

cd $(dirname $0)

find ../ -name "*.hpp" | xargs clang-format -i
find ../ -name "*.cpp" | xargs clang-format -i
