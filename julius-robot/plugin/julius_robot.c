#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "julius/juliuslib.h"
#include "plugin_defs.h"

#define PLUGIN_TITLE "Julius Robot"
#define MAX_RESULT_LEN 2048
#define CLASSIFIER_FILE "WPy64-31050\\python-3.10.5.amd64\\python classify.py"
#define CLASSIFIER_PARAM "--unit=500"

/* プロトタイプコール */
int startup(void *data);
int get_plugin_info(int opcode, char *buf, int buflen);
void output_result(Recog *recog, void *dummy);
void out_put_mfcc_csv(char *filename, float **mfcc, int len, int samplenum);
void out_put_data_csv(char *filename, float *mfcc, int len, float score, int frame, int word_num);
int read_setting_file();

/* scriptsフォルダが基準 */
#define CLASS_LABEL_FILE "../temp/_class.label"
#define PROJECT_NAME_FILE "../data/setting/word.conf"
#define TRAINING_SETTING_FILE "../data/setting/training.conf"
#define SELECT_MODE_FILE "../data/setting/mode.conf"
#define OK_WAV_LIST "../temp/_ok_wav.list"
#define BAD_WAV_LIST "../temp/_bad_wav.list"

int class_label;
char project_name[256];
int training_setting[7];
int select_mode;
int recog_count;
char wav_list[1024][256];
int max_recog;

/* スタートアップ関数. コールバックを登録する */
int startup(void *data) {
    Recog *recog = data;
    callback_add(recog, CALLBACK_RESULT, output_result, NULL);  
    setvbuf(stdout, 0, _IONBF, 0);

    read_setting_file();
    recog_count = 0;

    char buff[256];
    int i = 0;
    char basename[256];
    int c = '\\';
    char *ret;

    if (class_label == 0) {
        FILE *fp;
        fp = fopen(OK_WAV_LIST, "r");
        if (fp == NULL) {exit(1);}
        while (fgets(buff, 256, fp) != NULL) {
            ret = strrchr(buff, c);
            strncpy(basename, ret+1, strlen(ret)-6);
            basename[strlen(ret)-6] = '\0';
            strcpy(wav_list[i], basename);
            i += 1;
        }
        // max_recog = i;
        fclose(fp);
    }
    else {
        FILE *fp;
        fp = fopen(BAD_WAV_LIST, "r");
        if (fp == NULL) {exit(1);}
        while (fgets(buff, 256, fp) != NULL) {
            ret = strrchr(buff, c);
            strncpy(basename, ret+1, strlen(ret)-6);
            basename[strlen(ret)-6] = '\0';
            strcpy(wav_list[i], basename);
            i += 1;
        }
        // max_recog = i;
        fclose(fp);
    }
    return 0;
}

/* プラグイン情報を定義する (必ず必要) */
int get_plugin_info(int opcode, char *buf, int buflen) {
    switch(opcode) {
    case 0:
        strncpy(buf, PLUGIN_TITLE, buflen);
        break;
    }
    return 0;
}

/* 認識結果確定時に呼ばれる */
void output_result(Recog *recog, void *dummy) {
    int i, j;
    RecogProcess *r;
    WORD_INFO *winfo;
    WORD_ID *wid;
    Sentence *sent;
    int samplenum = recog->lmlist->am->mfcc->param->samplenum;
    float **mfcc = recog->lmlist->am->mfcc->param->parvec;
    int mfcclen = recog->lmlist->am->mfcc->param->veclen;
    float *avg_mfcc;
    char ph_prog_result[256], ph_result[MAX_RESULT_LEN], data_csv_name[256];
    char classification_cmd[256];


//////////////////////////////// 素性候補 /////////////
    char mfcc_csv_name[256];
    float score;
    int frame;
    int word_num;
//////////////////////////////////////////////////////


    /* 認識結果取得 */
    ph_result[0] = '\0';
    for(r=recog->process_list; r; r=r->next) {
        if(! r->live || r->result.status < 0) continue;

        /* 今回はベストの候補のみ取得する */
        winfo = r->lm->winfo;
        sent = &(r->result.sent[0]);
        wid = sent->word;


    //////////////////////////////// 素性候補 /////////////
        score = sent->score;
        frame = r->result.num_frame;
        word_num = sent->word_num;
    //////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////
        printf("Julius Mode: %d\n", select_mode);
        printf("Training Setting: %d%d%d%d%d%d%d\n", training_setting[0], training_setting[1], training_setting[2], training_setting[3], training_setting[4], training_setting[5], training_setting[6]);
        printf("Project Name: %s\n", project_name);
        printf("Class Label: %d\n", class_label);

        printf("Confidence: %f\n", sent->confidence[0]); // 素性 1 cmsscore -> 文字列ですよ！手直し必要
        printf("Score: %f\n", score); // 素性 2 文章スコア
        printf("Frame: %d\n", frame); // 素性 3 フレーム数
        printf("Word Num: %d\n", word_num);
    //////////////////////////////////////////////////////////

        for(i=0; i<sent->word_num; i++) { 
            for(j=0; j<winfo->wlen[wid[i]]; j++) {
                center_name(winfo->wseq[wid[i]][j]->name, ph_prog_result);
                sprintf(ph_result, "%s%s", ph_result, ph_prog_result);
            }
        }
    }

    /* 認識結果判定 */
    if(strstr(ph_result, project_name) != NULL) {
        printf("julius-chainer: %s recognized\n", project_name);
    }
    else {
        return;
    }

    // 後々 if mfcc で囲む
    /* mfcc 保存用配列の初期化 */
    avg_mfcc = (float *) malloc(sizeof(float) * mfcclen);

    for(i=0; i<mfcclen; i++) {
        avg_mfcc[i] = 0.0;
    }

    /* mfcc の瞬間値を発話平均化する */
    for(i=0; i<samplenum; i++) {
        for(j=0; j<mfcclen; j++) {
            avg_mfcc[j] += mfcc[i][j];
        }
    }
    for(i=0; i<mfcclen; i++) {
        avg_mfcc[i] /= samplenum;
    }

    /* 計算した 各素性 をファイルに出力 */
    if (class_label == 0) {
        if (training_setting[4] == 1) {
            sprintf(mfcc_csv_name, "../temp/mfcc_ok_%d_%s.csv", recog_count, wav_list[recog_count]);
            out_put_mfcc_csv(mfcc_csv_name, mfcc, mfcclen, samplenum);
        }
        if (training_setting[0] == 1 || training_setting[1] == 1 || training_setting[2] == 1 || training_setting[3] == 1) {
            sprintf(data_csv_name, "../temp/data_ok_%d_%s.csv", recog_count, wav_list[recog_count]);
            out_put_data_csv(data_csv_name, avg_mfcc, mfcclen, score, frame, word_num);
        }
    }
    else {
        if (training_setting[4] == 1) {
            sprintf(mfcc_csv_name, "../temp/mfcc_bad_%d_%s.csv", recog_count, wav_list[recog_count]);
            out_put_mfcc_csv(mfcc_csv_name, mfcc, mfcclen, samplenum);
        }
        if (training_setting[0] == 1 || training_setting[1] == 1 || training_setting[2] == 1 || training_setting[3] == 1) {
            sprintf(data_csv_name, "../temp/data_bad_%d_%s.csv", recog_count, wav_list[recog_count]);
            out_put_data_csv(data_csv_name, avg_mfcc, mfcclen, score, frame, word_num);
        }
    }

    // printf("Result print to: %s\n", data_csv_name);

    /* 機械学習の認識器を呼ぶ. フィードバック処理をここに書く */
    if (select_mode != 0) {
        sprintf(classification_cmd, CLASSIFIER_FILE " " CLASSIFIER_PARAM " --testfile=%s", data_csv_name);
        system(classification_cmd);

        remove(data_csv_name);
        remove(mfcc_csv_name);
    }
    recog_count++;
    free(avg_mfcc);
}

void out_put_mfcc_csv(char *filename, float **mfcc, int len, int samplenum) {

    FILE *fp;
    fp = fopen(filename, "w");
    if (fp == NULL) {exit(1);}

    /* 素性 MFCC をファイルに出力 */

    for (int i=0; i<samplenum; i++) {
        for (int j=0; j<len; j++) {
            fprintf(fp, "%.3f", mfcc[i][j]);
            if (j < len - 1) {
                fprintf(fp, ",");
            }
        }
        if (i < samplenum - 1) {
            fprintf(fp, "\n");
        }
    }
    fprintf(fp, "\n");
    fclose(fp);
}

/* 学習またはテストデータとして保存する */
void out_put_data_csv(char *filename, float *mfcc, int len, float score, int frame, int word_num) {

    int i;
    FILE *fp;
    fp = fopen(filename, "w");
    if (fp == NULL) {exit(1);}

    /* 先頭にクラスを追加 */
    fprintf(fp, "class:%d,", class_label);

    /* 以降に mfcc を追加 */
    for(i=0; i<len; i++) {
        fprintf(fp, "mfcc_avg:%.3f", mfcc[i]);
        if(i != len - 1) {
          fprintf(fp, ",");
        }
    }

    /* 素性 スコア を追加 */
    if (training_setting[1] == 1) {
        fprintf(fp, ",");
        fprintf(fp, "score:%.6f", score);
    }

    /* 素性 フレーム を追加 */
    if (training_setting[2] == 1) {
        fprintf(fp, ",");
        fprintf(fp, "frame:%d", frame);
    }

    /* 素性 文字数 を追加 */
    if (training_setting[3] == 1) {
        fprintf(fp, ",");
        fprintf(fp, "word_num:%d", word_num);
    }
   fprintf(fp, "\n");
   fclose(fp);
 }

/* 各種設定をファイルから取得 */
int read_setting_file() {

	FILE *fp1;
	char rc1;
	if ((fp1 = fopen(CLASS_LABEL_FILE, "r")) == NULL) {
		exit(EXIT_FAILURE);
	}
	if ((rc1 = fgetc(fp1)) != EOF) {
        class_label = rc1 - 48;
    }
	fclose(fp1);

    FILE *fp2;
    char s1[256];
    char s2[256];
    if ((fp2 = fopen(PROJECT_NAME_FILE, "r")) == NULL) {
        exit(EXIT_FAILURE);
    }
    while ((fscanf(fp2, "%[^,],%[^,]", s1, s2)) != EOF) {
        for (int i = 0; i < sizeof(s1) / sizeof(s1[0]); i++) {
            project_name[i] = s1[i];
          }
    }
    fclose(fp2);

    FILE *fp3;
    char rc3;
    if ((fp3 = fopen(TRAINING_SETTING_FILE, "r")) == NULL) {
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < sizeof(training_setting) / sizeof(training_setting[0]); i++) {
        if ((rc3 = fgetc(fp3)) != EOF) {
            training_setting[i] = rc3 - 48;
        }
    }
    fclose(fp3);

    FILE *fp4;
    char rc4;
    if ((fp4 = fopen(SELECT_MODE_FILE, "r")) == NULL) {
        exit(EXIT_FAILURE);
    }
    if ((rc4 = fgetc(fp4)) != EOF) {
        select_mode = rc4 - 48;
    }
    fclose(fp4);

    return 0;
}

/* end of file */
