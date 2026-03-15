# 正式编译（microgpt.c）
gcc -std=gnu17 -O2 -Wall -Wextra microgpt.c -lm -o task

# 调试编译（microgpt.c）
gcc -std=gnu17 -O0 -g -Wall -Wextra -fsanitize=address,undefined microgpt.c -lm -o task
