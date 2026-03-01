#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MAX_DOCS 1000000
#define MAX_LEN 256
typedef struct Value Value;
char *docs[MAX_DOCS];
int num_docs = 0;
char uchars[256];
int vocab_size = 0;
int BOS = 0;
int charset[256] = {0};
//确定是否有训练语料
int file_exists(const char *filename){
    FILE *f = fopen(filename, "r");
    if(f){
        fclose(f);
        return 1;
    }
    return 0;
}
//没有训练语料就自动下载
void download_file(){
    system(
        "curl -L "
        "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt "
        "-o input.txt"
    );
}
//将训练语料写入docs
void load_docs(){
    FILE *f = fopen("input.txt", "r");
    if (!f){
            perror("open failed");
            return;
        }
    char buffer[MAX_LEN];

    while(fgets(buffer, MAX_LEN, f)){
        
        buffer[strcspn(buffer,"\n")] = 0;

        if(strlen(buffer)==0)
            continue;

        docs[num_docs] = strdup(buffer);
        num_docs++;
    }
    fclose(f);
}
//将语料随机打乱
void shuffle_docs(){
    for(int i = num_docs-1; i > 0 ; i--){
        int j = rand() % (i+1);

        char *tmp = docs[i];
        docs[i] = docs[j];
        docs[j] = tmp;
    }
}
//构建tokenizer第一步:构建无重复词表
void build_vocab(){
    for(int i= 0;i < num_docs; i++){
        char *s = docs[i];
        for(int j=0 ; s[j] ; j++){
            unsigned char c = (unsigned char)s[j];//unsigned 可将char范围变为[0,255], 避免了负数.
            charset[c] = 1;
        }
    }
    vocab_size = 0;
    for(int i = 0; i < 256 ; i++){
        if(charset[i]){
            uchars[vocab_size++] = (char)i;
        }
    }
    BOS = vocab_size;
    vocab_size += 1;
}
//开始实现自动求导系统(计算图 | 自动微分 | 反向传播)
typedef struct Value{
    double data;
    double grad;
    struct Value **children;
    double *local_grads;
    int n_children;
} Value;
//创建节点
Value* craet_value(
    double data,
    Value **children,
    double *local_grads,
    int n_children
){
    Value *v = malloc(sizeof(Value));
    v->data = data;
    v->gard = 0;
    v->children = children;
    v->local_grads = local_grads;
    v->n_children = n_children;
    
    return v;
}
Value* add(Value *a,Value *b){
    Value **children = malloc(2*sizeof(Value*));
    double *gards = malloc(2*sizeof(double));

    children[0] = a;
    children[1] = b;

    grads[0] = 1.0;
    grads[1] = 1.0;

    return create_value(
        
    )
}








int main(){
    srand(42);
    if(!file_exists("input.txt")){
        download_file();
    }
    load_docs();
    shuffle_docs();
    printf("num docs: %d\n", num_docs);
    build_vocab();
    printf("vocab size: %d\n", vocab_size);
    for (int i = 0; i < num_docs; i++) {
        free(docs[i]);
    }
    return 0;
}
