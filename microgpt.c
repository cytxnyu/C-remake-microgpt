#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MAX_DOCS 1000000
#define MAX_LEN 256
char *docs[MAX_DOCS];
int num_docs = 0;

int file_exists(const char *filename){
    FILE *f = fopen(filename, "r");
    if(f){
        fclose(f);
        return 1;
    }
    return 0;
}
//if training is expected, open it.Otherwise,download it automatically. 
void download_file(){
    system(
        "curl -L "
        "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt "
        "-o input.txt"
    );
}
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
void shuffle_docs(){
    for(int i = num_docs-1; i > 0 ; i--){
        int j = rand() % (i+1);

        char *tmp = docs[i];
        docs[i] = docs[j];
        docs[j] = tmp;
    }
}
int main(){
    srand(42);
    if(!file_exists("input.txt")){
        download_file();
    }
    load_docs();
    shuffle_docs();
    printf("num docs: %d\n", num_docs);
    for (int i = 0; i < num_docs; i++) {
        free(docs[i]);
    }
    return 0;
}
