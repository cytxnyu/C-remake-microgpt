#正式编译
gcc -std=gnu17 -O2 -Wall -Wextra -lm microgpt.c -o microgpt

#调试编译
gcc -std=gnu17 -O0 -g -Wall -Wextra -fsanitize=address,undefined -lm microgpt.c -o microgpt