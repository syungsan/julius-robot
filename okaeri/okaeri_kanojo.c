#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "julius/juliuslib.h"
#include "plugin_defs.h"

#define PLUGIN_TITLE "Okaeri Kanojo"
#define MAX_RESULT_LEN 2048
#define ALWAYS_OUT_MFCC_TO_FILE 0 /* 特徴量ファイルを常に作成するか. 1 にすると話者認識が行われない */
#define CLASSIFIER_FILE "python classify.py"
#define CLASSIFIER_PARAM "--unit=500"
#define OKAERI_CLASS 0

/* プロトタイプコール */
int startup(void *data);
int get_plugin_info(int opcode, char *buf, int buflen);
void output_result(Recog *recog, void *dummy);
void output_test_data_csv(char *filename, int cls, float *mfcc, int len);

/* スタートアップ関数. コールバックを登録する */
int startup(void *data) {
  Recog *recog = data;
  callback_add(recog, CALLBACK_RESULT, output_result, NULL);  
  setvbuf(stdout, 0, _IONBF, 0);
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
  time_t dt;
  struct tm *dt_data;
  float **mfcc = recog->lmlist->am->mfcc->param->parvec;
  int mfcclen = recog->lmlist->am->mfcc->param->veclen;
  float *avg_mfcc;
  char ph_prog_result[256], ph_result[MAX_RESULT_LEN], test_csv_name[256];
#if ALWAYS_OUT_MFCC_TO_FILE != 1
  char classification_cmd[256];
#endif
  
  /* 認識結果取得 */
  ph_result[0] = '\0';
  for(r=recog->process_list; r; r=r->next) {
    if(! r->live || r->result.status < 0) continue;

    /* 今回はベストの候補のみ取得する */
    winfo = r->lm->winfo;
    sent = &(r->result.sent[0]);
    wid = sent->word;

    for(i=0; i<sent->word_num; i++) { 
      for(j=0; j<winfo->wlen[wid[i]]; j++) {
        center_name(winfo->wseq[wid[i]][j]->name, ph_prog_result);
        sprintf(ph_result, "%s%s", ph_result, ph_prog_result);
      }
    }
  }

  /* 認識結果判定 */
  if(strstr(ph_result, "tadaima") != NULL) {
    printf("okaeri_kanojo: tadaima recognized\n");
  }
#if ALWAYS_OUT_MFCC_TO_FILE != 1
  else {
    /* 常に mfcc を出力しない設定であればこれで終了 */
    return;
  }
#endif

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

  /* 計算した mfcc をファイルに出力 */
  dt = time(NULL);
  dt_data = localtime(&dt);
  sprintf(test_csv_name, "test/test_%04d%02d%02d_%02d%02d%02d.csv", dt_data->tm_year + 1900, dt_data->tm_mon + 1, dt_data->tm_mday,
		  dt_data->tm_hour, dt_data->tm_min, dt_data->tm_sec);
  output_test_data_csv(test_csv_name, OKAERI_CLASS, avg_mfcc, mfcclen);

  /* 話者認識器を呼ぶ. おかえり音声もここで再生される
   * 余談: 本当はプラグイン側で音声再生させたかったが余力がなく TODO */
#if ALWAYS_OUT_MFCC_TO_FILE != 1
  sprintf(classification_cmd, CLASSIFIER_FILE " " CLASSIFIER_PARAM " --testfile=%s", test_csv_name);
  system(classification_cmd);

  remove(test_csv_name);
#endif
  free(avg_mfcc);
}

/* テストデータとして保存する */
void output_test_data_csv(char *filename, int cls, float *mfcc, int len) {
  FILE *fp;
  int i;

  fp = fopen(filename, "w");
  if(fp == NULL) { exit(1); }

  /* 先頭にクラスを追加 */
  fprintf(fp, "%d,", cls);

  /* 以降に mfcc を追加 */
  for(i=0; i<len; i++) {
    fprintf(fp, "%.3f", mfcc[i]);
    if(i != len - 1) {
      fprintf(fp, ",");
    }
  }
  fprintf(fp, "\n");
  fclose(fp);
}

/* end of file */
